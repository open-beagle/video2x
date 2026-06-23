from __future__ import annotations

import argparse
import ctypes
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cu
from cuda.bindings import runtime as cudart


KERNEL = r"""
extern "C" __global__
void rgb8_to_chw_float(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = y * width + x;
    int src_idx = pixel * 3;
    int plane = width * height;
    dst[pixel] = (float)src[src_idx + 0] * (1.0f / 255.0f);
    dst[plane + pixel] = (float)src[src_idx + 1] * (1.0f / 255.0f);
    dst[2 * plane + pixel] = (float)src[src_idx + 2] * (1.0f / 255.0f);
}

static __device__ __forceinline__
unsigned short float_to_half_bits(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    unsigned int mant = x & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return (unsigned short)sign;
        }
        mant = (mant | 0x00800000u) >> (1 - exp);
        return (unsigned short)(sign | ((mant + 0x00001000u) >> 13));
    }
    if (exp >= 31) {
        return (unsigned short)(sign | 0x7c00u);
    }
    return (unsigned short)(sign | ((unsigned int)exp << 10) | ((mant + 0x00001000u) >> 13));
}

extern "C" __global__
void rgb8_to_chw_half(
    const unsigned char* __restrict__ src,
    unsigned short* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = y * width + x;
    int src_idx = pixel * 3;
    int plane = width * height;
    dst[pixel] = float_to_half_bits((float)src[src_idx + 0] * (1.0f / 255.0f));
    dst[plane + pixel] = float_to_half_bits((float)src[src_idx + 1] * (1.0f / 255.0f));
    dst[2 * plane + pixel] = float_to_half_bits((float)src[src_idx + 2] * (1.0f / 255.0f));
}

extern "C" __global__
void chw_float_to_rgb8_resize_pad(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int dst_idx = (y * dst_w + x) * 3;
    if (x < pad_left || x >= pad_left + content_w) {
        dst[dst_idx + 0] = 0;
        dst[dst_idx + 1] = 0;
        dst[dst_idx + 2] = 0;
        return;
    }

    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y1 >= src_h) y1 = src_h - 1;

    int plane = src_w * src_h;
    for (int c = 0; c < 3; ++c) {
        const float* p = src + c * plane;
        float v00 = p[y0 * src_w + x0];
        float v01 = p[y0 * src_w + x1];
        float v10 = p[y1 * src_w + x0];
        float v11 = p[y1 * src_w + x1];
        float v0 = v00 + (v01 - v00) * fx;
        float v1 = v10 + (v11 - v10) * fx;
        float v = v0 + (v1 - v0) * fy;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        dst[dst_idx + c] = (unsigned char)(v * 255.0f + 0.5f);
    }
}

static __device__ __forceinline__
float half_bits_to_float(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int exp = ((unsigned int)h & 0x7c00u) >> 10;
    unsigned int mant = (unsigned int)h & 0x03ffu;
    unsigned int out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    return __uint_as_float(out);
}

extern "C" __global__
void chw_half_to_rgb8_resize_pad(
    const unsigned short* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int dst_idx = (y * dst_w + x) * 3;
    if (x < pad_left || x >= pad_left + content_w) {
        dst[dst_idx + 0] = 0;
        dst[dst_idx + 1] = 0;
        dst[dst_idx + 2] = 0;
        return;
    }

    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y1 >= src_h) y1 = src_h - 1;

    int plane = src_w * src_h;
    for (int c = 0; c < 3; ++c) {
        const unsigned short* p = src + c * plane;
        float v00 = half_bits_to_float(p[y0 * src_w + x0]);
        float v01 = half_bits_to_float(p[y0 * src_w + x1]);
        float v10 = half_bits_to_float(p[y1 * src_w + x0]);
        float v11 = half_bits_to_float(p[y1 * src_w + x1]);
        float v0 = v00 + (v01 - v00) * fx;
        float v1 = v10 + (v11 - v10) * fx;
        float v = v0 + (v1 - v0) * fy;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        dst[dst_idx + c] = (unsigned char)(v * 255.0f + 0.5f);
    }
}

static __device__ __forceinline__
float sample_chw_pixel(
    const float* __restrict__ src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    int c
) {
    if (x < pad_left || x >= pad_left + content_w) {
        return 0.0f;
    }

    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y1 >= src_h) y1 = src_h - 1;

    int plane = src_w * src_h;
    const float* p = src + c * plane;
    float v00 = p[y0 * src_w + x0];
    float v01 = p[y0 * src_w + x1];
    float v10 = p[y1 * src_w + x0];
    float v11 = p[y1 * src_w + x1];
    float v0 = v00 + (v01 - v00) * fx;
    float v1 = v10 + (v11 - v10) * fx;
    float v = v0 + (v1 - v0) * fy;
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

static __device__ __forceinline__
float sample_chw_half_pixel(
    const unsigned short* __restrict__ src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    int c
) {
    if (x < pad_left || x >= pad_left + content_w) {
        return 0.0f;
    }

    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y1 >= src_h) y1 = src_h - 1;

    int plane = src_w * src_h;
    const unsigned short* p = src + c * plane;
    float v00 = half_bits_to_float(p[y0 * src_w + x0]);
    float v01 = half_bits_to_float(p[y0 * src_w + x1]);
    float v10 = half_bits_to_float(p[y1 * src_w + x0]);
    float v11 = half_bits_to_float(p[y1 * src_w + x1]);
    float v0 = v00 + (v01 - v00) * fx;
    float v1 = v10 + (v11 - v10) * fx;
    float v = v0 + (v1 - v0) * fy;
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

static __device__ __forceinline__
unsigned char clamp_u8(float v) {
    v = fminf(fmaxf(v, 0.0f), 255.0f);
    return (unsigned char)(v + 0.5f);
}

extern "C" __global__
void chw_half_to_nv12_resize_pad(
    const unsigned short* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float r = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    dst[y * dst_w + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = dst_w * dst_h + (y / 2) * dst_w + x;
        dst[uv_idx] = clamp_u8(u_sum * 0.25f);
        dst[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
    }
}

extern "C" __global__
void chw_half_to_nv12_resize_pad_pitched(
    const unsigned short* __restrict__ src,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float r = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = (y / 2) * uv_pitch + x;
        uv_plane[uv_idx] = clamp_u8(u_sum * 0.25f);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
    }
}

extern "C" __global__
void chw_float_to_nv12_resize_pad(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float r = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    dst[y * dst_w + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = dst_w * dst_h + (y / 2) * dst_w + x;
        dst[uv_idx] = clamp_u8(u_sum * 0.25f);
        dst[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
    }
}

extern "C" __global__
void chw_float_to_nv12_resize_pad_pitched(
    const float* __restrict__ src,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float r = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = (y / 2) * uv_pitch + x;
        uv_plane[uv_idx] = clamp_u8(u_sum * 0.25f);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
    }
}

static __device__ __forceinline__
float half_chw_value(
    const unsigned short* __restrict__ src,
    int c,
    int y,
    int x,
    int height,
    int width
) {
    return half_bits_to_float(src[(c * height + y) * width + x]);
}

static __device__ __forceinline__
float srvgg_tail_hr_pixel(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    int lr_w,
    int lr_h,
    int hx,
    int hy,
    int c
) {
    hx = min(max(hx, 0), lr_w * 4 - 1);
    hy = min(max(hy, 0), lr_h * 4 - 1);
    int lx = hx >> 2;
    int ly = hy >> 2;
    int sub_x = hx & 3;
    int sub_y = hy & 3;
    int out_ch = c * 16 + sub_y * 4 + sub_x;

    float sum = half_bits_to_float(bias[out_ch]);
    for (int ic = 0; ic < 64; ++ic) {
        for (int ky = 0; ky < 3; ++ky) {
            int fy = ly + ky - 1;
            if (fy < 0 || fy >= lr_h) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int fx = lx + kx - 1;
                if (fx < 0 || fx >= lr_w) continue;
                int widx = ((out_ch * 64 + ic) * 3 + ky) * 3 + kx;
                sum += half_chw_value(feature, ic, fy, fx, lr_h, lr_w) * half_bits_to_float(weight[widx]);
            }
        }
    }
    sum += half_chw_value(input, c, ly, lx, lr_h, lr_w);
    return sum;
}

static __device__ __forceinline__
void srvgg_tail_hr_rgb(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    int lr_w,
    int lr_h,
    int hx,
    int hy,
    float* r,
    float* g,
    float* b
) {
    hx = min(max(hx, 0), lr_w * 4 - 1);
    hy = min(max(hy, 0), lr_h * 4 - 1);
    int lx = hx >> 2;
    int ly = hy >> 2;
    int sub_x = hx & 3;
    int sub_y = hy & 3;
    int sub = sub_y * 4 + sub_x;
    int out_ch_r = sub;
    int out_ch_g = 16 + sub;
    int out_ch_b = 32 + sub;

    float sum_r = half_bits_to_float(bias[out_ch_r]);
    float sum_g = half_bits_to_float(bias[out_ch_g]);
    float sum_b = half_bits_to_float(bias[out_ch_b]);
    for (int ic = 0; ic < 64; ++ic) {
        for (int ky = 0; ky < 3; ++ky) {
            int fy = ly + ky - 1;
            if (fy < 0 || fy >= lr_h) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int fx = lx + kx - 1;
                if (fx < 0 || fx >= lr_w) continue;
                float f = half_chw_value(feature, ic, fy, fx, lr_h, lr_w);
                int base = (ic * 3 + ky) * 3 + kx;
                sum_r += f * half_bits_to_float(weight[out_ch_r * 64 * 9 + base]);
                sum_g += f * half_bits_to_float(weight[out_ch_g * 64 * 9 + base]);
                sum_b += f * half_bits_to_float(weight[out_ch_b * 64 * 9 + base]);
            }
        }
    }
    *r = sum_r + half_chw_value(input, 0, ly, lx, lr_h, lr_w);
    *g = sum_g + half_chw_value(input, 1, ly, lx, lr_h, lr_w);
    *b = sum_b + half_chw_value(input, 2, ly, lx, lr_h, lr_w);
}

static __device__ __forceinline__
float sample_srvgg_tail_pixel(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    int c
) {
    if (x < pad_left || x >= pad_left + content_w) {
        return 0.0f;
    }

    int src_w = lr_w * 4;
    int src_h = lr_h * 4;
    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float v00 = srvgg_tail_hr_pixel(feature, input, weight, bias, lr_w, lr_h, x0, y0, c);
    float v01 = srvgg_tail_hr_pixel(feature, input, weight, bias, lr_w, lr_h, x1, y0, c);
    float v10 = srvgg_tail_hr_pixel(feature, input, weight, bias, lr_w, lr_h, x0, y1, c);
    float v11 = srvgg_tail_hr_pixel(feature, input, weight, bias, lr_w, lr_h, x1, y1, c);
    float v0 = v00 + (v01 - v00) * fx;
    float v1 = v10 + (v11 - v10) * fx;
    return fminf(fmaxf(v0 + (v1 - v0) * fy, 0.0f), 1.0f);
}

static __device__ __forceinline__
void sample_srvgg_tail_rgb_pixel(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    float* r,
    float* g,
    float* b
) {
    if (x < pad_left || x >= pad_left + content_w) {
        *r = 0.0f;
        *g = 0.0f;
        *b = 0.0f;
        return;
    }

    int src_w = lr_w * 4;
    int src_h = lr_h * 4;
    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
    srvgg_tail_hr_rgb(feature, input, weight, bias, lr_w, lr_h, x0, y0, &r00, &g00, &b00);
    srvgg_tail_hr_rgb(feature, input, weight, bias, lr_w, lr_h, x1, y0, &r01, &g01, &b01);
    srvgg_tail_hr_rgb(feature, input, weight, bias, lr_w, lr_h, x0, y1, &r10, &g10, &b10);
    srvgg_tail_hr_rgb(feature, input, weight, bias, lr_w, lr_h, x1, y1, &r11, &g11, &b11);

    float r0 = r00 + (r01 - r00) * fx;
    float r1 = r10 + (r11 - r10) * fx;
    float g0 = g00 + (g01 - g00) * fx;
    float g1 = g10 + (g11 - g10) * fx;
    float b0 = b00 + (b01 - b00) * fx;
    float b1 = b10 + (b11 - b10) * fx;
    *r = fminf(fmaxf(r0 + (r1 - r0) * fy, 0.0f), 1.0f);
    *g = fminf(fmaxf(g0 + (g1 - g0) * fy, 0.0f), 1.0f);
    *b = fminf(fmaxf(b0 + (b1 - b0) * fy, 0.0f), 1.0f);
}

static __device__ __forceinline__
void sample_srvgg_tail_rgb_pixel_cell_fused(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    float* r,
    float* g,
    float* b
) {
    if (x < pad_left || x >= pad_left + content_w) {
        *r = 0.0f;
        *g = 0.0f;
        *b = 0.0f;
        return;
    }

    int src_w = lr_w * 4;
    int src_h = lr_h * 4;
    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    int lx0 = x0 >> 2;
    int ly0 = y0 >> 2;
    int lx1 = x1 >> 2;
    int ly1 = y1 >> 2;

    if (lx0 != lx1 || ly0 != ly1) {
        sample_srvgg_tail_rgb_pixel(
            feature, input, weight, bias,
            lr_w, lr_h, dst_w, dst_h, content_w, pad_left,
            x, y, r, g, b
        );
        return;
    }

    int sub00 = (y0 & 3) * 4 + (x0 & 3);
    int sub01 = (y0 & 3) * 4 + (x1 & 3);
    int sub10 = (y1 & 3) * 4 + (x0 & 3);
    int sub11 = (y1 & 3) * 4 + (x1 & 3);
    float c00 = (1.0f - fx) * (1.0f - fy);
    float c01 = fx * (1.0f - fy);
    float c10 = (1.0f - fx) * fy;
    float c11 = fx * fy;

    int r00 = sub00;
    int r01 = sub01;
    int r10 = sub10;
    int r11 = sub11;
    int g00 = 16 + sub00;
    int g01 = 16 + sub01;
    int g10 = 16 + sub10;
    int g11 = 16 + sub11;
    int b00 = 32 + sub00;
    int b01 = 32 + sub01;
    int b10 = 32 + sub10;
    int b11 = 32 + sub11;

    float sum_r =
        c00 * half_bits_to_float(bias[r00]) +
        c01 * half_bits_to_float(bias[r01]) +
        c10 * half_bits_to_float(bias[r10]) +
        c11 * half_bits_to_float(bias[r11]);
    float sum_g =
        c00 * half_bits_to_float(bias[g00]) +
        c01 * half_bits_to_float(bias[g01]) +
        c10 * half_bits_to_float(bias[g10]) +
        c11 * half_bits_to_float(bias[g11]);
    float sum_b =
        c00 * half_bits_to_float(bias[b00]) +
        c01 * half_bits_to_float(bias[b01]) +
        c10 * half_bits_to_float(bias[b10]) +
        c11 * half_bits_to_float(bias[b11]);

    for (int ic = 0; ic < 64; ++ic) {
        for (int ky = 0; ky < 3; ++ky) {
            int fy0 = ly0 + ky - 1;
            if (fy0 < 0 || fy0 >= lr_h) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int fx0 = lx0 + kx - 1;
                if (fx0 < 0 || fx0 >= lr_w) continue;
                float f = half_chw_value(feature, ic, fy0, fx0, lr_h, lr_w);
                int base = (ic * 3 + ky) * 3 + kx;
                float wr =
                    c00 * half_bits_to_float(weight[r00 * 64 * 9 + base]) +
                    c01 * half_bits_to_float(weight[r01 * 64 * 9 + base]) +
                    c10 * half_bits_to_float(weight[r10 * 64 * 9 + base]) +
                    c11 * half_bits_to_float(weight[r11 * 64 * 9 + base]);
                float wg =
                    c00 * half_bits_to_float(weight[g00 * 64 * 9 + base]) +
                    c01 * half_bits_to_float(weight[g01 * 64 * 9 + base]) +
                    c10 * half_bits_to_float(weight[g10 * 64 * 9 + base]) +
                    c11 * half_bits_to_float(weight[g11 * 64 * 9 + base]);
                float wb =
                    c00 * half_bits_to_float(weight[b00 * 64 * 9 + base]) +
                    c01 * half_bits_to_float(weight[b01 * 64 * 9 + base]) +
                    c10 * half_bits_to_float(weight[b10 * 64 * 9 + base]) +
                    c11 * half_bits_to_float(weight[b11 * 64 * 9 + base]);
                sum_r += f * wr;
                sum_g += f * wg;
                sum_b += f * wb;
            }
        }
    }

    sum_r += half_chw_value(input, 0, ly0, lx0, lr_h, lr_w);
    sum_g += half_chw_value(input, 1, ly0, lx0, lr_h, lr_w);
    sum_b += half_chw_value(input, 2, ly0, lx0, lr_h, lr_w);
    *r = fminf(fmaxf(sum_r, 0.0f), 1.0f);
    *g = fminf(fmaxf(sum_g, 0.0f), 1.0f);
    *b = fminf(fmaxf(sum_b, 0.0f), 1.0f);
}

static __device__ __forceinline__
void srvgg_conv48_hr_rgb(
    const unsigned short* __restrict__ conv48,
    const unsigned short* __restrict__ input,
    int lr_w,
    int lr_h,
    int hx,
    int hy,
    float* r,
    float* g,
    float* b
) {
    hx = min(max(hx, 0), lr_w * 4 - 1);
    hy = min(max(hy, 0), lr_h * 4 - 1);
    int lx = hx >> 2;
    int ly = hy >> 2;
    int sub = (hy & 3) * 4 + (hx & 3);
    *r = half_chw_value(conv48, sub, ly, lx, lr_h, lr_w) + half_chw_value(input, 0, ly, lx, lr_h, lr_w);
    *g = half_chw_value(conv48, 16 + sub, ly, lx, lr_h, lr_w) + half_chw_value(input, 1, ly, lx, lr_h, lr_w);
    *b = half_chw_value(conv48, 32 + sub, ly, lx, lr_h, lr_w) + half_chw_value(input, 2, ly, lx, lr_h, lr_w);
}

static __device__ __forceinline__
void sample_srvgg_conv48_rgb_pixel(
    const unsigned short* __restrict__ conv48,
    const unsigned short* __restrict__ input,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    float* r,
    float* g,
    float* b
) {
    if (x < pad_left || x >= pad_left + content_w) {
        *r = 0.0f;
        *g = 0.0f;
        *b = 0.0f;
        return;
    }

    int src_w = lr_w * 4;
    int src_h = lr_h * 4;
    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
    srvgg_conv48_hr_rgb(conv48, input, lr_w, lr_h, x0, y0, &r00, &g00, &b00);
    srvgg_conv48_hr_rgb(conv48, input, lr_w, lr_h, x1, y0, &r01, &g01, &b01);
    srvgg_conv48_hr_rgb(conv48, input, lr_w, lr_h, x0, y1, &r10, &g10, &b10);
    srvgg_conv48_hr_rgb(conv48, input, lr_w, lr_h, x1, y1, &r11, &g11, &b11);

    float r0 = r00 + (r01 - r00) * fx;
    float r1 = r10 + (r11 - r10) * fx;
    float g0 = g00 + (g01 - g00) * fx;
    float g1 = g10 + (g11 - g10) * fx;
    float b0 = b00 + (b01 - b00) * fx;
    float b1 = b10 + (b11 - b10) * fx;
    *r = fminf(fmaxf(r0 + (r1 - r0) * fy, 0.0f), 1.0f);
    *g = fminf(fmaxf(g0 + (g1 - g0) * fy, 0.0f), 1.0f);
    *b = fminf(fmaxf(b0 + (b1 - b0) * fy, 0.0f), 1.0f);
}

extern "C" __global__
void srvgg_tail_half_to_nv12_resize_pad_pitched(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float r = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = (y / 2) * uv_pitch + x;
        uv_plane[uv_idx] = clamp_u8(u_sum * 0.25f);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
    }
}

extern "C" __global__
void srvgg_tail_half_to_nv12_resize_pad_pitched_2x2(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = bx * 2;
    int y0 = by * 2;
    if (x0 >= dst_w || y0 >= dst_h) return;

    float u_sum = 0.0f;
    float v_sum = 0.0f;
    int samples = 0;

    for (int oy = 0; oy < 2; ++oy) {
        int y = y0 + oy;
        if (y >= dst_h) continue;
        for (int ox = 0; ox < 2; ++ox) {
            int x = x0 + ox;
            if (x >= dst_w) continue;

            float r = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
            float g = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
            float b = sample_srvgg_tail_pixel(feature, input, weight, bias, lr_w, lr_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
            y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);
            u_sum += -0.148f * r - 0.291f * g + 0.439f * b + 128.0f;
            v_sum += 0.439f * r - 0.368f * g - 0.071f * b + 128.0f;
            samples += 1;
        }
    }

    if (samples > 0) {
        float inv = 1.0f / (float)samples;
        int uv_idx = (y0 / 2) * uv_pitch + x0;
        uv_plane[uv_idx] = clamp_u8(u_sum * inv);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * inv);
    }
}

extern "C" __global__
void srvgg_tail_half_to_nv12_resize_pad_pitched_2x2_rgb(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = bx * 2;
    int y0 = by * 2;
    if (x0 >= dst_w || y0 >= dst_h) return;

    float u_sum = 0.0f;
    float v_sum = 0.0f;
    int samples = 0;

    for (int oy = 0; oy < 2; ++oy) {
        int y = y0 + oy;
        if (y >= dst_h) continue;
        for (int ox = 0; ox < 2; ++ox) {
            int x = x0 + ox;
            if (x >= dst_w) continue;

            float r01, g01, b01;
            sample_srvgg_tail_rgb_pixel(
                feature, input, weight, bias,
                lr_w, lr_h, dst_w, dst_h, content_w, pad_left,
                x, y, &r01, &g01, &b01
            );
            float r = r01 * 255.0f;
            float g = g01 * 255.0f;
            float b = b01 * 255.0f;
            y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);
            u_sum += -0.148f * r - 0.291f * g + 0.439f * b + 128.0f;
            v_sum += 0.439f * r - 0.368f * g - 0.071f * b + 128.0f;
            samples += 1;
        }
    }

    if (samples > 0) {
        float inv = 1.0f / (float)samples;
        int uv_idx = (y0 / 2) * uv_pitch + x0;
        uv_plane[uv_idx] = clamp_u8(u_sum * inv);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * inv);
    }
}

extern "C" __global__
void srvgg_tail_half_to_nv12_resize_pad_pitched_2x2_rgb_cell(
    const unsigned short* __restrict__ feature,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = bx * 2;
    int y0 = by * 2;
    if (x0 >= dst_w || y0 >= dst_h) return;

    float u_sum = 0.0f;
    float v_sum = 0.0f;
    int samples = 0;

    for (int oy = 0; oy < 2; ++oy) {
        int y = y0 + oy;
        if (y >= dst_h) continue;
        for (int ox = 0; ox < 2; ++ox) {
            int x = x0 + ox;
            if (x >= dst_w) continue;

            float r01, g01, b01;
            sample_srvgg_tail_rgb_pixel_cell_fused(
                feature, input, weight, bias,
                lr_w, lr_h, dst_w, dst_h, content_w, pad_left,
                x, y, &r01, &g01, &b01
            );
            float r = r01 * 255.0f;
            float g = g01 * 255.0f;
            float b = b01 * 255.0f;
            y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);
            u_sum += -0.148f * r - 0.291f * g + 0.439f * b + 128.0f;
            v_sum += 0.439f * r - 0.368f * g - 0.071f * b + 128.0f;
            samples += 1;
        }
    }

    if (samples > 0) {
        float inv = 1.0f / (float)samples;
        int uv_idx = (y0 / 2) * uv_pitch + x0;
        uv_plane[uv_idx] = clamp_u8(u_sum * inv);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * inv);
    }
}

extern "C" __global__
void srvgg_conv48_to_nv12_resize_pad_pitched_2x2_rgb(
    const unsigned short* __restrict__ conv48,
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ bias,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int lr_w,
    int lr_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = bx * 2;
    int y0 = by * 2;
    if (x0 >= dst_w || y0 >= dst_h) return;

    float u_sum = 0.0f;
    float v_sum = 0.0f;
    int samples = 0;

    for (int oy = 0; oy < 2; ++oy) {
        int y = y0 + oy;
        if (y >= dst_h) continue;
        for (int ox = 0; ox < 2; ++ox) {
            int x = x0 + ox;
            if (x >= dst_w) continue;

            float r01, g01, b01;
            sample_srvgg_conv48_rgb_pixel(
                conv48, input,
                lr_w, lr_h, dst_w, dst_h, content_w, pad_left,
                x, y, &r01, &g01, &b01
            );
            float r = r01 * 255.0f;
            float g = g01 * 255.0f;
            float b = b01 * 255.0f;
            y_plane[y * y_pitch + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);
            u_sum += -0.148f * r - 0.291f * g + 0.439f * b + 128.0f;
            v_sum += 0.439f * r - 0.368f * g - 0.071f * b + 128.0f;
            samples += 1;
        }
    }

    if (samples > 0) {
        float inv = 1.0f / (float)samples;
        int uv_idx = (y0 / 2) * uv_pitch + x0;
        uv_plane[uv_idx] = clamp_u8(u_sum * inv);
        uv_plane[uv_idx + 1] = clamp_u8(v_sum * inv);
    }
}
"""


def check_cuda(result, label: str):
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{label} failed: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def check_driver(result, label: str):
    err = result[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label} failed: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def alloc_pinned_buffer(size: int, label: str) -> tuple[int, np.ndarray, memoryview]:
    ptr = check_cuda(cudart.cudaHostAlloc(size, cudart.cudaHostAllocDefault), f"cudaHostAlloc {label}")
    array_type = ctypes.c_uint8 * size
    array = np.ctypeslib.as_array(array_type.from_address(int(ptr)))
    return int(ptr), array, memoryview(array)


class CudaP010Bridge:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"CUDA P010 bridge not found: {path}")
        self.lib = ctypes.CDLL(str(path))
        self.lib.open_p010_decoder.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.open_p010_decoder.restype = ctypes.c_void_p
        self.lib.decode_next_p010_to_chw_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.decode_next_p010_to_chw_fp16.restype = ctypes.c_int
        self.lib.close_p010_decoder.argtypes = [ctypes.c_void_p]
        self.lib.close_p010_decoder.restype = None
        self.errbuf = ctypes.create_string_buffer(2048)

    def open(self, input_path: Path, width: int, height: int) -> int:
        handle = self.lib.open_p010_decoder(str(input_path).encode(), width, height, self.errbuf, len(self.errbuf))
        if not handle:
            raise RuntimeError(self.errbuf.value.decode(errors="replace"))
        return int(handle)

    def decode_next(self, handle: int, d_input: int) -> int:
        result = self.lib.decode_next_p010_to_chw_fp16(
            ctypes.c_void_p(handle),
            ctypes.c_void_p(int(d_input)),
            self.errbuf,
            len(self.errbuf),
        )
        if result < 0:
            raise RuntimeError(self.errbuf.value.decode(errors="replace"))
        return int(result)

    def close(self, handle: int) -> None:
        self.lib.close_p010_decoder(ctypes.c_void_p(handle))


class CudaNvencBridge:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"CUDA NVENC bridge not found: {path}")
        self.lib = ctypes.CDLL(str(path))
        self.lib.open_cuda_nvenc_writer.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.open_cuda_nvenc_writer.restype = ctypes.c_void_p
        self.lib.begin_cuda_nvenc_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.begin_cuda_nvenc_frame.restype = ctypes.c_void_p
        self.lib.send_cuda_nvenc_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.send_cuda_nvenc_frame.restype = ctypes.c_int
        self.lib.close_cuda_nvenc_writer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.close_cuda_nvenc_writer.restype = ctypes.c_int
        self.errbuf = ctypes.create_string_buffer(2048)

    def open(self, output_path: Path, width: int, height: int, fps: int, bitrate: str, gop_size: int) -> int:
        handle = self.lib.open_cuda_nvenc_writer(
            str(output_path).encode(),
            width,
            height,
            fps,
            bitrate.encode(),
            gop_size,
            self.errbuf,
            len(self.errbuf),
        )
        if not handle:
            raise RuntimeError(self.errbuf.value.decode(errors="replace"))
        return int(handle)

    def begin_frame(self, handle: int) -> tuple[int, int, int, int, int]:
        y_device = ctypes.c_void_p()
        uv_device = ctypes.c_void_p()
        y_pitch = ctypes.c_int()
        uv_pitch = ctypes.c_int()
        frame = self.lib.begin_cuda_nvenc_frame(
            ctypes.c_void_p(handle),
            ctypes.byref(y_device),
            ctypes.byref(uv_device),
            ctypes.byref(y_pitch),
            ctypes.byref(uv_pitch),
            self.errbuf,
            len(self.errbuf),
        )
        if not frame:
            raise RuntimeError(self.errbuf.value.decode(errors="replace"))
        return int(frame), int(y_device.value), int(uv_device.value), int(y_pitch.value), int(uv_pitch.value)

    def send_frame(self, handle: int, frame: int, pts: int) -> None:
        result = self.lib.send_cuda_nvenc_frame(
            ctypes.c_void_p(handle),
            ctypes.c_void_p(frame),
            ctypes.c_int64(pts),
            self.errbuf,
            len(self.errbuf),
        )
        if result < 0:
            raise RuntimeError(self.errbuf.value.decode(errors="replace"))

    def close(self, handle: int) -> None:
        result = self.lib.close_cuda_nvenc_writer(ctypes.c_void_p(handle), self.errbuf, len(self.errbuf))
        if result < 0:
            raise RuntimeError(self.errbuf.value.decode(errors="replace"))


@dataclass
class PipelineSlot:
    index: int
    stream: object
    context: object
    raw_host_ptr: int
    raw_host: np.ndarray
    raw_host_view: memoryview
    frame_host_ptr: int
    host_frame: np.ndarray
    host_frame_view: memoryview
    d_raw: int
    d_input: int
    d_output: int
    d_frame: int
    nvenc_frame: int = 0
    pending: bool = False
    frame_index: int = 0


def dtype_to_np(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT:
        return np.float32
    if dtype == trt.DataType.HALF:
        return np.float16
    raise ValueError(f"unsupported dtype: {dtype}")


def tensor_nbytes(shape: tuple[int, ...], dtype: trt.DataType) -> int:
    return math.prod(shape) * np.dtype(dtype_to_np(dtype)).itemsize


def load_srvgg_tail_weights(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    weight = np.ascontiguousarray(data["weight"].astype(np.float16, copy=False).reshape(-1))
    bias = np.ascontiguousarray(data["bias"].astype(np.float16, copy=False).reshape(-1))
    if weight.size != 48 * 64 * 3 * 3:
        raise RuntimeError(f"unexpected SRVGG tail weight size {weight.size}: {path}")
    if bias.size != 48:
        raise RuntimeError(f"unexpected SRVGG tail bias size {bias.size}: {path}")
    return weight, bias


def kernel_ptx_path() -> Path:
    return Path(__file__).resolve().with_name("postprocess.ptx")


def compile_kernel(load_preprocess: bool) -> dict[tuple[str, str], cu.CUfunction]:
    ptx_path = kernel_ptx_path()
    if not ptx_path.exists():
        raise RuntimeError(f"precompiled CUDA kernel not found: {ptx_path}")
    module = check_driver(cu.cuModuleLoadData(ptx_path.read_bytes()), "cuModuleLoadData")
    kernels = {
        ("float", "rgb24"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_float_to_rgb8_resize_pad"),
            "cuModuleGetFunction float rgb",
        ),
        ("float", "nv12"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_float_to_nv12_resize_pad"),
            "cuModuleGetFunction float nv12",
        ),
        ("float", "nv12-pitched"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_float_to_nv12_resize_pad_pitched"),
            "cuModuleGetFunction float nv12 pitched",
        ),
        ("half", "rgb24"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_half_to_rgb8_resize_pad"),
            "cuModuleGetFunction half rgb",
        ),
        ("half", "nv12"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_half_to_nv12_resize_pad"),
            "cuModuleGetFunction half nv12",
        ),
        ("half", "nv12-pitched"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_half_to_nv12_resize_pad_pitched"),
            "cuModuleGetFunction half nv12 pitched",
        ),
    }
    if load_preprocess:
        kernels[("preprocess", "float")] = check_driver(
            cu.cuModuleGetFunction(module, b"rgb8_to_chw_float"),
            "cuModuleGetFunction preprocess float",
        )
        kernels[("preprocess", "half")] = check_driver(
            cu.cuModuleGetFunction(module, b"rgb8_to_chw_half"),
            "cuModuleGetFunction preprocess half",
        )
    kernels[("srvgg-tail", "half-nv12-pitched")] = check_driver(
        cu.cuModuleGetFunction(module, b"srvgg_tail_half_to_nv12_resize_pad_pitched"),
        "cuModuleGetFunction srvgg tail half nv12 pitched",
    )
    kernels[("srvgg-tail", "half-nv12-pitched-2x2")] = check_driver(
        cu.cuModuleGetFunction(module, b"srvgg_tail_half_to_nv12_resize_pad_pitched_2x2"),
        "cuModuleGetFunction srvgg tail half nv12 pitched 2x2",
    )
    kernels[("srvgg-tail", "half-nv12-pitched-2x2-rgb")] = check_driver(
        cu.cuModuleGetFunction(module, b"srvgg_tail_half_to_nv12_resize_pad_pitched_2x2_rgb"),
        "cuModuleGetFunction srvgg tail half nv12 pitched 2x2 rgb",
    )
    kernels[("srvgg-tail", "half-nv12-pitched-2x2-rgb-cell")] = check_driver(
        cu.cuModuleGetFunction(module, b"srvgg_tail_half_to_nv12_resize_pad_pitched_2x2_rgb_cell"),
        "cuModuleGetFunction srvgg tail half nv12 pitched 2x2 rgb cell",
    )
    kernels[("srvgg-conv48-tail", "half-nv12-pitched-2x2-rgb")] = check_driver(
        cu.cuModuleGetFunction(module, b"srvgg_conv48_to_nv12_resize_pad_pitched_2x2_rgb"),
        "cuModuleGetFunction srvgg conv48 tail half nv12 pitched 2x2 rgb",
    )
    return kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real video TRT + CUDA postprocess runner.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=720)
    parser.add_argument("--input-height", type=int, default=420)
    parser.add_argument("--decode-width", type=int, default=0)
    parser.add_argument("--decode-height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument("--content-width", type=int, default=1852)
    parser.add_argument("--encoder", default="libx265")
    parser.add_argument("--bitrate", default="5M")
    parser.add_argument("--gop-size", type=int, default=0)
    parser.add_argument("--output-pix-fmt", choices=["rgb24", "nv12"], default="rgb24")
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--input-mode", choices=["rgb24", "cuda-p010"], default="rgb24")
    parser.add_argument("--cuda-p010-bridge", type=Path, default=Path("/app/src/libffmpeg_cuda_chw_bridge.so"))
    parser.add_argument("--output-mode", choices=["stdin", "cuda-nvenc"], default="stdin")
    parser.add_argument("--cuda-nvenc-bridge", type=Path, default=Path("/app/src/libffmpeg_cuda_chw_bridge.so"))
    parser.add_argument(
        "--postprocess-mode",
        choices=["engine-output", "srvgg-fused-tail", "srvgg-conv48-tail"],
        default="engine-output",
    )
    parser.add_argument("--srvgg-tail-weights", type=Path, default=None)
    parser.add_argument(
        "--srvgg-tail-kernel",
        choices=["pixel", "2x2", "2x2-rgb", "2x2-rgb-cell"],
        default="2x2-rgb",
    )
    return parser.parse_args()


def doubled_bitrate(value: str) -> str:
    if len(value) > 1 and value[-1].isalpha() and value[:-1].isdigit():
        return f"{int(value[:-1]) * 2}{value[-1]}"
    if value.isdigit():
        return str(int(value) * 2)
    return value


def merge_audio(input_path: Path, video_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    check_driver(cu.cuInit(0), "cuInit")
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    kernels = compile_kernel(args.input_mode == "rgb24")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to load engine: {args.engine}")
    input_name = output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name
    assert input_name and output_name

    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    input_dtype = engine.get_tensor_dtype(input_name)
    output_dtype = engine.get_tensor_dtype(output_name)
    if args.input_mode == "cuda-p010" and input_dtype != trt.DataType.HALF:
        raise RuntimeError(f"cuda-p010 input mode requires FP16 engine input, got {input_dtype}")
    if args.output_mode == "cuda-nvenc":
        if args.encoder != "hevc_nvenc":
            raise RuntimeError("cuda-nvenc output mode currently requires --encoder hevc_nvenc")
        if args.output_pix_fmt != "nv12":
            raise RuntimeError("cuda-nvenc output mode currently requires --output-pix-fmt nv12")
    if args.postprocess_mode in {"srvgg-fused-tail", "srvgg-conv48-tail"}:
        if args.postprocess_mode == "srvgg-fused-tail" and args.input_mode != "cuda-p010":
            raise RuntimeError("srvgg-fused-tail currently requires --input-mode cuda-p010")
        if input_dtype != trt.DataType.HALF or output_dtype != trt.DataType.HALF:
            raise RuntimeError(f"{args.postprocess_mode} requires FP16 input and FP16 output")
    decode_width = args.decode_width or args.input_width
    decode_height = args.decode_height or args.input_height
    if input_shape != (1, 3, decode_height, decode_width):
        raise RuntimeError(f"engine input shape {input_shape} does not match decode size {decode_width}x{decode_height}")
    if args.postprocess_mode == "srvgg-fused-tail" and output_shape != (1, 64, decode_height, decode_width):
        raise RuntimeError(
            f"srvgg-fused-tail expects feature output shape {(1, 64, decode_height, decode_width)}, got {output_shape}"
        )
    if args.postprocess_mode == "srvgg-conv48-tail" and output_shape != (1, 48, decode_height, decode_width):
        raise RuntimeError(
            f"srvgg-conv48-tail expects conv output shape {(1, 48, decode_height, decode_width)}, got {output_shape}"
        )
    input_kind = "half" if input_dtype == trt.DataType.HALF else "float"
    output_kind = "half" if output_dtype == trt.DataType.HALF else "float"

    input_bytes = tensor_nbytes(input_shape, input_dtype)
    output_bytes = tensor_nbytes(output_shape, output_dtype)
    frame_bytes = args.target_width * args.target_height * (3 if args.output_pix_fmt == "rgb24" else 3 // 2)
    if args.output_pix_fmt == "nv12":
        frame_bytes = args.target_width * args.target_height * 3 // 2
    raw_in_bytes = decode_width * decode_height * 3
    src_h = output_shape[2]
    src_w = output_shape[3]
    pad_left = (args.target_width - args.content_width) // 2
    pipeline_depth = max(1, args.pipeline_depth)
    bridge = CudaP010Bridge(args.cuda_p010_bridge) if args.input_mode == "cuda-p010" else None
    nvenc_bridge = CudaNvencBridge(args.cuda_nvenc_bridge) if args.output_mode == "cuda-nvenc" else None
    d_tail_weight = d_tail_bias = 0
    if args.postprocess_mode == "srvgg-fused-tail":
        assert args.srvgg_tail_weights is not None
        tail_weight, tail_bias = load_srvgg_tail_weights(args.srvgg_tail_weights)
        d_tail_weight = check_cuda(cudart.cudaMalloc(tail_weight.nbytes), "cudaMalloc srvgg tail weight")
        d_tail_bias = check_cuda(cudart.cudaMalloc(tail_bias.nbytes), "cudaMalloc srvgg tail bias")
        check_cuda(
            cudart.cudaMemcpy(
                d_tail_weight,
                tail_weight.ctypes.data,
                tail_weight.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "copy srvgg tail weight",
        )
        check_cuda(
            cudart.cudaMemcpy(
                d_tail_bias,
                tail_bias.ctypes.data,
                tail_bias.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "copy srvgg tail bias",
        )

    slots: list[PipelineSlot] = []
    for index in range(pipeline_depth):
        stream = check_cuda(cudart.cudaStreamCreate(), f"cudaStreamCreate slot {index}")
        slot_context = engine.create_execution_context()
        if args.input_mode == "rgb24":
            raw_host_ptr, raw_host, raw_host_view = alloc_pinned_buffer(raw_in_bytes, f"raw input slot {index}")
            d_raw = check_cuda(cudart.cudaMalloc(raw_in_bytes), f"cudaMalloc raw input slot {index}")
        else:
            raw_host_ptr = 0
            raw_host = np.empty(0, dtype=np.uint8)
            raw_host_view = memoryview(raw_host)
            d_raw = 0
        if args.output_mode == "stdin":
            frame_host_ptr, host_frame, host_frame_view = alloc_pinned_buffer(frame_bytes, f"frame output slot {index}")
        else:
            frame_host_ptr = 0
            host_frame = np.empty(0, dtype=np.uint8)
            host_frame_view = memoryview(host_frame)
        d_input = check_cuda(cudart.cudaMalloc(input_bytes), f"cudaMalloc input slot {index}")
        d_output = check_cuda(cudart.cudaMalloc(output_bytes), f"cudaMalloc output slot {index}")
        d_frame = check_cuda(cudart.cudaMalloc(frame_bytes), f"cudaMalloc frame slot {index}") if args.output_mode == "stdin" else 0
        slot_context.set_tensor_address(input_name, int(d_input))
        slot_context.set_tensor_address(output_name, int(d_output))
        slots.append(
            PipelineSlot(
                index=index,
                stream=stream,
                context=slot_context,
                raw_host_ptr=raw_host_ptr,
                raw_host=raw_host,
                raw_host_view=raw_host_view,
                frame_host_ptr=frame_host_ptr,
                host_frame=host_frame,
                host_frame_view=host_frame_view,
                d_raw=d_raw,
                d_input=d_input,
                d_output=d_output,
                d_frame=d_frame,
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        no_audio = tmpdir / "video_no_audio.mp4"
        encode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            args.output_pix_fmt,
            "-s",
            f"{args.target_width}x{args.target_height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            args.encoder,
        ]
        if args.gop_size > 0:
            encode_cmd += ["-g", str(args.gop_size)]
        if args.bitrate:
            encode_cmd += ["-b:v", args.bitrate, "-maxrate", args.bitrate, "-bufsize", doubled_bitrate(args.bitrate)]
        elif args.encoder in {"libx264", "libx265"}:
            encode_cmd += ["-crf", "18"]
        if args.encoder in {"libx264", "libx265"}:
            encode_cmd += ["-preset", "ultrafast"]
            if args.encoder == "libx265":
                encode_cmd += ["-x265-params", "log-level=error"]
        elif args.encoder in {"h264_nvenc", "hevc_nvenc"}:
            encode_cmd += ["-preset", "p1", "-tune", "ull"]
            if args.gop_size > 0:
                encode_cmd += ["-forced-idr", "1"]
        encode_cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(no_audio)]

        decoder = None
        bridge_handle = None
        if args.input_mode == "rgb24":
            decode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(args.input),
                "-map",
                "0:v:0",
            ]
            if decode_width != args.input_width or decode_height != args.input_height:
                decode_cmd += ["-vf", f"scale={decode_width}:{decode_height}:flags=bicubic"]
            decode_cmd += [
                "-nostdin",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ]
            decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            assert bridge is not None
            bridge_handle = bridge.open(args.input, decode_width, decode_height)
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE) if args.output_mode == "stdin" else None
        nvenc_handle = (
            nvenc_bridge.open(no_audio, args.target_width, args.target_height, args.fps, args.bitrate, args.gop_size)
            if nvenc_bridge
            else None
        )
        started = time.time()
        frames = 0
        submitted = 0
        completed = 0
        decode_time = preprocess_time = h2d_time = infer_time = kernel_time = d2h_time = sync_time = encode_time = 0.0
        frame_limit = args.frames or args.expected_frames
        last_progress_time = time.time()
        last_progress_frames = 0
        progress_interval = int(os.environ.get("PROGRESS_INTERVAL", "30"))

        def submit_slot(slot: PipelineSlot) -> None:
            nonlocal h2d_time, preprocess_time, infer_time, kernel_time, d2h_time, submitted

            if args.input_mode == "rgb24":
                t0 = time.perf_counter()
                check_cuda(
                    cudart.cudaMemcpyAsync(
                        slot.d_raw,
                        slot.raw_host.ctypes.data,
                        raw_in_bytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        slot.stream,
                    ),
                    f"H2D slot {slot.index}",
                )
                h2d_time += time.perf_counter() - t0

                preprocess_params = [
                    ctypes.c_void_p(int(slot.d_raw)),
                    ctypes.c_void_p(int(slot.d_input)),
                    ctypes.c_int(decode_width),
                    ctypes.c_int(decode_height),
                ]
                preprocess_param_ptrs = (ctypes.c_void_p * len(preprocess_params))(
                    *[ctypes.addressof(p) for p in preprocess_params]
                )

                t0 = time.perf_counter()
                check_driver(
                    cu.cuLaunchKernel(
                        kernels[("preprocess", input_kind)],
                        math.ceil(decode_width / 16),
                        math.ceil(decode_height / 16),
                        1,
                        16,
                        16,
                        1,
                        0,
                        int(slot.stream),
                        preprocess_param_ptrs,
                        0,
                    ),
                    f"cuLaunchKernel preprocess slot {slot.index}",
                )
                preprocess_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            if not slot.context.execute_async_v3(int(slot.stream)):
                raise RuntimeError(f"execute_async_v3 failed slot {slot.index}")
            infer_time += time.perf_counter() - t0

            if args.output_mode == "cuda-nvenc":
                assert nvenc_bridge is not None and nvenc_handle is not None
                frame_handle, y_device, uv_device, y_pitch, uv_pitch = nvenc_bridge.begin_frame(nvenc_handle)
                slot.nvenc_frame = frame_handle
                if args.postprocess_mode in {"srvgg-fused-tail", "srvgg-conv48-tail"}:
                    params = [
                        ctypes.c_void_p(int(slot.d_output)),
                        ctypes.c_void_p(int(slot.d_input)),
                        ctypes.c_void_p(int(d_tail_weight)),
                        ctypes.c_void_p(int(d_tail_bias)),
                        ctypes.c_void_p(y_device),
                        ctypes.c_void_p(uv_device),
                        ctypes.c_int(y_pitch),
                        ctypes.c_int(uv_pitch),
                        ctypes.c_int(decode_width),
                        ctypes.c_int(decode_height),
                        ctypes.c_int(args.target_width),
                        ctypes.c_int(args.target_height),
                        ctypes.c_int(args.content_width),
                        ctypes.c_int(pad_left),
                    ]
                    if args.postprocess_mode == "srvgg-conv48-tail":
                        postprocess_key = ("srvgg-conv48-tail", "half-nv12-pitched-2x2-rgb")
                    elif args.srvgg_tail_kernel == "2x2-rgb-cell":
                        postprocess_key = ("srvgg-tail", "half-nv12-pitched-2x2-rgb-cell")
                    elif args.srvgg_tail_kernel == "2x2-rgb":
                        postprocess_key = ("srvgg-tail", "half-nv12-pitched-2x2-rgb")
                    elif args.srvgg_tail_kernel == "2x2":
                        postprocess_key = ("srvgg-tail", "half-nv12-pitched-2x2")
                    else:
                        postprocess_key = ("srvgg-tail", "half-nv12-pitched")
                else:
                    params = [
                        ctypes.c_void_p(int(slot.d_output)),
                        ctypes.c_void_p(y_device),
                        ctypes.c_void_p(uv_device),
                        ctypes.c_int(y_pitch),
                        ctypes.c_int(uv_pitch),
                        ctypes.c_int(src_w),
                        ctypes.c_int(src_h),
                        ctypes.c_int(args.target_width),
                        ctypes.c_int(args.target_height),
                        ctypes.c_int(args.content_width),
                        ctypes.c_int(pad_left),
                    ]
                    postprocess_key = (output_kind, "nv12-pitched")
            else:
                params = [
                    ctypes.c_void_p(int(slot.d_output)),
                    ctypes.c_void_p(int(slot.d_frame)),
                    ctypes.c_int(src_w),
                    ctypes.c_int(src_h),
                    ctypes.c_int(args.target_width),
                    ctypes.c_int(args.target_height),
                    ctypes.c_int(args.content_width),
                    ctypes.c_int(pad_left),
                ]
                postprocess_key = (output_kind, args.output_pix_fmt)
            param_ptrs = (ctypes.c_void_p * len(params))(*[ctypes.addressof(p) for p in params])

            t0 = time.perf_counter()
            grid_pixel_scale = (
                2
                if postprocess_key[1]
                in {
                    "half-nv12-pitched-2x2",
                    "half-nv12-pitched-2x2-rgb",
                    "half-nv12-pitched-2x2-rgb-cell",
                }
                else 1
            )
            grid_w_pixels = math.ceil(args.target_width / grid_pixel_scale)
            grid_h_pixels = math.ceil(args.target_height / grid_pixel_scale)
            check_driver(
                cu.cuLaunchKernel(
                    kernels[postprocess_key],
                    math.ceil(grid_w_pixels / 16),
                    math.ceil(grid_h_pixels / 16),
                    1,
                    16,
                    16,
                    1,
                    0,
                    int(slot.stream),
                    param_ptrs,
                    0,
                ),
                f"cuLaunchKernel postprocess slot {slot.index}",
            )
            kernel_time += time.perf_counter() - t0

            if args.output_mode == "stdin":
                t0 = time.perf_counter()
                check_cuda(
                    cudart.cudaMemcpyAsync(
                        slot.host_frame.ctypes.data,
                        slot.d_frame,
                        frame_bytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        slot.stream,
                    ),
                    f"D2H frame slot {slot.index}",
                )
                d2h_time += time.perf_counter() - t0

            slot.pending = True
            slot.frame_index = submitted
            submitted += 1

        def finish_slot(slot: PipelineSlot) -> None:
            nonlocal sync_time, encode_time, frames, completed, last_progress_time, last_progress_frames
            if not slot.pending:
                return
            t0 = time.perf_counter()
            check_cuda(cudart.cudaStreamSynchronize(slot.stream), f"sync slot {slot.index}")
            sync_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            if args.output_mode == "cuda-nvenc":
                assert nvenc_bridge is not None and nvenc_handle is not None
                nvenc_bridge.send_frame(nvenc_handle, slot.nvenc_frame, slot.frame_index)
                slot.nvenc_frame = 0
            else:
                assert encoder is not None
                if not encoder.stdin:
                    raise RuntimeError("encoder stdin closed")
                encoder.stdin.write(slot.host_frame_view)
            encode_time += time.perf_counter() - t0
            slot.pending = False
            frames += 1
            completed += 1

            now = time.time()
            if now - last_progress_time >= progress_interval:
                elapsed_interval = now - last_progress_time
                frames_interval = frames - last_progress_frames
                fps_interval = frames_interval / elapsed_interval if elapsed_interval > 0 else 0
                pct = (frames / frame_limit * 100) if frame_limit else 0
                eta_str = ""
                if fps_interval > 0 and frame_limit:
                    eta_sec = (frame_limit - frames) / fps_interval
                    eta_str = f", ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}"
                print(
                    f"Progress: {frames}/{frame_limit} ({pct:.1f}%), Speed: {fps_interval:.2f} fps{eta_str}",
                    flush=True,
                )
                last_progress_time = now
                last_progress_frames = frames

        try:
            while True:
                if frame_limit and submitted >= frame_limit:
                    break
                slot = slots[submitted % pipeline_depth]
                if slot.pending:
                    finish_slot(slot)
                t0 = time.perf_counter()
                if args.input_mode == "rgb24":
                    read_bytes = decoder.stdout.readinto(slot.raw_host_view) if decoder and decoder.stdout else 0
                    decode_time += time.perf_counter() - t0
                    if not read_bytes:
                        break
                    if read_bytes != raw_in_bytes:
                        raise RuntimeError(f"incomplete input frame: {read_bytes} != {raw_in_bytes}")
                else:
                    assert bridge is not None and bridge_handle is not None
                    decoded = bridge.decode_next(bridge_handle, int(slot.d_input))
                    decode_time += time.perf_counter() - t0
                    if decoded == 0:
                        break
                    preprocess_time += 0.0
                    read_bytes = raw_in_bytes
                submit_slot(slot)

            while completed < submitted:
                finish_slot(slots[completed % pipeline_depth])
        finally:
            if decoder and decoder.stdout:
                decoder.stdout.close()
            if encoder and encoder.stdin:
                encoder.stdin.close()
            if decoder:
                decoder.wait()
            if bridge and bridge_handle is not None:
                bridge.close(bridge_handle)
            if nvenc_bridge and nvenc_handle is not None:
                nvenc_bridge.close(nvenc_handle)
            if encoder:
                encoder.wait()

        if encoder and encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {encoder.returncode}")
        if decoder and decoder.returncode != 0 and not (frame_limit and frames >= frame_limit):
            raise RuntimeError(f"ffmpeg decode failed: {decoder.returncode}")

        merge_started = time.perf_counter()
        merge_audio(args.input, no_audio, args.output)
        merge_time = time.perf_counter() - merge_started
        elapsed = max(time.time() - started, 0.001)

    print(f"frames={frames} elapsed={elapsed:.3f}s fps={frames / elapsed:.3f}")
    print(
        f"decode={decode_time:.3f}s preprocess={preprocess_time:.3f}s h2d={h2d_time:.3f}s "
        f"infer={infer_time:.3f}s kernel={kernel_time:.3f}s d2h_frame={d2h_time:.3f}s sync_wait={sync_time:.3f}s "
        f"encode_write={encode_time:.3f}s merge_audio={merge_time:.3f}s"
    )

    for slot in slots:
        if slot.d_raw:
            check_cuda(cudart.cudaFree(slot.d_raw), f"free raw input slot {slot.index}")
        check_cuda(cudart.cudaFree(slot.d_input), f"free input slot {slot.index}")
        check_cuda(cudart.cudaFree(slot.d_output), f"free output slot {slot.index}")
        if slot.d_frame:
            check_cuda(cudart.cudaFree(slot.d_frame), f"free frame slot {slot.index}")
        if slot.raw_host_ptr:
            check_cuda(cudart.cudaFreeHost(slot.raw_host_ptr), f"free raw host slot {slot.index}")
        if slot.frame_host_ptr:
            check_cuda(cudart.cudaFreeHost(slot.frame_host_ptr), f"free frame host slot {slot.index}")
        check_cuda(cudart.cudaStreamDestroy(slot.stream), f"destroy stream slot {slot.index}")
    if d_tail_weight:
        check_cuda(cudart.cudaFree(d_tail_weight), "free srvgg tail weight")
    if d_tail_bias:
        check_cuda(cudart.cudaFree(d_tail_bias), "free srvgg tail bias")
    if not args.output.exists():
        shutil.copy2(no_audio, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
