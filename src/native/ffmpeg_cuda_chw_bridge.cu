#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_CUDA;

static void set_error(char *errbuf, int errbuf_size, const char *fmt, ...) {
    if (!errbuf || errbuf_size <= 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(errbuf, (size_t)errbuf_size, fmt, args);
    va_end(args);
}

static void set_av_error(char *errbuf, int errbuf_size, const char *what, int err) {
    char avbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, avbuf, sizeof(avbuf));
    set_error(errbuf, errbuf_size, "%s: %s", what, avbuf);
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    (void)ctx;
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == hw_pix_fmt) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

static __device__ __forceinline__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __device__ __forceinline__ unsigned short float_to_half_bits(float f) {
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

static __device__ __forceinline__
void p010_to_rgb01(
    const unsigned char *__restrict__ y_plane,
    const unsigned char *__restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int x,
    int y,
    float *r,
    float *g,
    float *b
) {
    const unsigned short *y_row = (const unsigned short *)(y_plane + y * y_pitch);
    const unsigned short *uv_row = (const unsigned short *)(uv_plane + (y / 2) * uv_pitch);
    int uv_x = (x / 2) * 2;

    float yy = ((float)((y_row[x] >> 6) - 64)) * (1.0f / 876.0f);
    float uu = ((float)((uv_row[uv_x + 0] >> 6) - 512)) * (1.0f / 896.0f);
    float vv = ((float)((uv_row[uv_x + 1] >> 6) - 512)) * (1.0f / 896.0f);

    *r = clamp01(yy + 1.5748f * vv);
    *g = clamp01(yy - 0.1873f * uu - 0.4681f * vv);
    *b = clamp01(yy + 1.8556f * uu);
}

static __device__ __forceinline__
void p010_to_rgb01_bilinear(
    const unsigned char *__restrict__ y_plane,
    const unsigned char *__restrict__ uv_plane,
    int y_pitch,
    int uv_pitch,
    int width,
    int height,
    float sx,
    float sy,
    float *r,
    float *g,
    float *b
) {
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) {
        x0 = 0;
        fx = 0.0f;
    }
    if (y0 < 0) {
        y0 = 0;
        fy = 0.0f;
    }
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
    p010_to_rgb01(y_plane, uv_plane, y_pitch, uv_pitch, x0, y0, &r00, &g00, &b00);
    p010_to_rgb01(y_plane, uv_plane, y_pitch, uv_pitch, x1, y0, &r01, &g01, &b01);
    p010_to_rgb01(y_plane, uv_plane, y_pitch, uv_pitch, x0, y1, &r10, &g10, &b10);
    p010_to_rgb01(y_plane, uv_plane, y_pitch, uv_pitch, x1, y1, &r11, &g11, &b11);

    float r0 = r00 + (r01 - r00) * fx;
    float r1 = r10 + (r11 - r10) * fx;
    float g0 = g00 + (g01 - g00) * fx;
    float g1 = g10 + (g11 - g10) * fx;
    float b0 = b00 + (b01 - b00) * fx;
    float b1 = b10 + (b11 - b10) * fx;
    *r = clamp01(r0 + (r1 - r0) * fy);
    *g = clamp01(g0 + (g1 - g0) * fy);
    *b = clamp01(b0 + (b1 - b0) * fy);
}

extern "C" __global__
void p010_to_chw_half_bridge_kernel(
    const unsigned char *__restrict__ y_plane,
    const unsigned char *__restrict__ uv_plane,
    unsigned short *__restrict__ dst,
    int y_pitch,
    int uv_pitch,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) {
        return;
    }

    float r;
    float g;
    float b;
    if (src_width == dst_width && src_height == dst_height) {
        p010_to_rgb01(y_plane, uv_plane, y_pitch, uv_pitch, x, y, &r, &g, &b);
    } else {
        float sx = ((float)x + 0.5f) * ((float)src_width / (float)dst_width) - 0.5f;
        float sy = ((float)y + 0.5f) * ((float)src_height / (float)dst_height) - 0.5f;
        p010_to_rgb01_bilinear(
            y_plane,
            uv_plane,
            y_pitch,
            uv_pitch,
            src_width,
            src_height,
            sx,
            sy,
            &r,
            &g,
            &b
        );
    }

    int pixel = y * dst_width + x;
    int plane = dst_width * dst_height;
    dst[pixel] = float_to_half_bits(r);
    dst[plane + pixel] = float_to_half_bits(g);
    dst[2 * plane + pixel] = float_to_half_bits(b);
}

static int create_primary_cuda_hwdevice(AVBufferRef **hw_device_ctx, char *errbuf, int errbuf_size) {
    CUresult cuerr = cuInit(0);
    if (cuerr != CUDA_SUCCESS) {
        set_error(errbuf, errbuf_size, "cuInit failed: %d", (int)cuerr);
        return -1;
    }

    CUdevice device = 0;
    cuerr = cuDeviceGet(&device, 0);
    if (cuerr != CUDA_SUCCESS) {
        set_error(errbuf, errbuf_size, "cuDeviceGet failed: %d", (int)cuerr);
        return -1;
    }

    CUcontext primary = NULL;
    cuerr = cuDevicePrimaryCtxRetain(&primary, device);
    if (cuerr != CUDA_SUCCESS) {
        set_error(errbuf, errbuf_size, "cuDevicePrimaryCtxRetain failed: %d", (int)cuerr);
        return -1;
    }

    cuerr = cuCtxSetCurrent(primary);
    if (cuerr != CUDA_SUCCESS) {
        set_error(errbuf, errbuf_size, "cuCtxSetCurrent failed: %d", (int)cuerr);
        cuDevicePrimaryCtxRelease(device);
        return -1;
    }

    AVBufferRef *device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!device_ref) {
        set_error(errbuf, errbuf_size, "av_hwdevice_ctx_alloc cuda failed");
        cuDevicePrimaryCtxRelease(device);
        return -1;
    }

    AVHWDeviceContext *device_ctx = (AVHWDeviceContext *)device_ref->data;
    AVCUDADeviceContext *cuda_device = (AVCUDADeviceContext *)device_ctx->hwctx;
    cuda_device->cuda_ctx = primary;

    int err = av_hwdevice_ctx_init(device_ref);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "av_hwdevice_ctx_init cuda primary", err);
        av_buffer_unref(&device_ref);
        cuDevicePrimaryCtxRelease(device);
        return -1;
    }

    *hw_device_ctx = device_ref;
    return 0;
}

struct BridgeDecoder {
    AVFormatContext *fmt;
    AVCodecContext *ctx;
    AVBufferRef *hw_device_ctx;
    AVPacket *packet;
    AVFrame *frame;
    cudaStream_t stream;
    int video_stream;
    int width;
    int height;
    int sent_eof;
};

struct BridgeNvencWriter {
    AVFormatContext *fmt;
    AVCodecContext *enc;
    AVBufferRef *hw_device_ctx;
    AVBufferRef *frames_ctx;
    int stream_index;
    int width;
    int height;
    int fps;
};

struct BridgeNvencFrame {
    AVFrame *frame;
};

static int64_t parse_bitrate(const char *value) {
    if (!value || !*value) {
        return 5000000;
    }
    char *end = NULL;
    double number = strtod(value, &end);
    if (number <= 0.0) {
        return 5000000;
    }
    double multiplier = 1.0;
    if (end && *end) {
        if (*end == 'k' || *end == 'K') {
            multiplier = 1000.0;
        } else if (*end == 'm' || *end == 'M') {
            multiplier = 1000000.0;
        } else if (*end == 'g' || *end == 'G') {
            multiplier = 1000000000.0;
        }
    }
    return (int64_t)(number * multiplier);
}

static int write_nvenc_packets(BridgeNvencWriter *writer, char *errbuf, int errbuf_size) {
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        set_error(errbuf, errbuf_size, "av_packet_alloc failed");
        return -1;
    }
    while (1) {
        int err = avcodec_receive_packet(writer->enc, pkt);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
            break;
        }
        if (err < 0) {
            set_av_error(errbuf, errbuf_size, "avcodec_receive_packet", err);
            av_packet_free(&pkt);
            return -1;
        }
        pkt->stream_index = writer->stream_index;
        av_packet_rescale_ts(pkt, writer->enc->time_base, writer->fmt->streams[writer->stream_index]->time_base);
        err = av_interleaved_write_frame(writer->fmt, pkt);
        if (err < 0) {
            set_av_error(errbuf, errbuf_size, "av_interleaved_write_frame", err);
            av_packet_free(&pkt);
            return -1;
        }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    return 0;
}

static int convert_frame_to_chw_fp16(
    const AVFrame *frame,
    void *dst_device,
    int expected_width,
    int expected_height,
    cudaStream_t stream,
    char *errbuf,
    int errbuf_size
) {
    AVHWFramesContext *frames = frame->hw_frames_ctx ? (AVHWFramesContext *)frame->hw_frames_ctx->data : NULL;
    if ((enum AVPixelFormat)frame->format != AV_PIX_FMT_CUDA || !frames || frames->sw_format != AV_PIX_FMT_P010LE) {
        set_error(errbuf, errbuf_size, "expected cuda/p010le frame, got format=%d", frame->format);
        return -1;
    }
    dim3 block(16, 16);
    dim3 grid((expected_width + block.x - 1) / block.x, (expected_height + block.y - 1) / block.y);
    p010_to_chw_half_bridge_kernel<<<grid, block, 0, stream>>>(
        frame->data[0],
        frame->data[1],
        (unsigned short *)dst_device,
        frame->linesize[0],
        frame->linesize[1],
        frame->width,
        frame->height,
        expected_width,
        expected_height
    );
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        set_error(errbuf, errbuf_size, "p010_to_chw_half kernel failed: %s", cudaGetErrorString(cuda_err));
        return -1;
    }
    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        set_error(errbuf, errbuf_size, "cudaStreamSynchronize bridge failed: %s", cudaGetErrorString(cuda_err));
        return -1;
    }
    return 0;
}

extern "C"
BridgeDecoder *open_p010_decoder(
    const char *input,
    int expected_width,
    int expected_height,
    char *errbuf,
    int errbuf_size
) {
    if (!input) {
        set_error(errbuf, errbuf_size, "input is required");
        return NULL;
    }

    BridgeDecoder *decoder = new BridgeDecoder();
    memset(decoder, 0, sizeof(*decoder));
    decoder->video_stream = -1;
    decoder->width = expected_width;
    decoder->height = expected_height;
    AVStream *stream = NULL;
    const AVCodec *codec = NULL;

    int err = avformat_open_input(&decoder->fmt, input, NULL, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avformat_open_input", err);
        goto fail;
    }
    err = avformat_find_stream_info(decoder->fmt, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avformat_find_stream_info", err);
        goto fail;
    }

    decoder->video_stream = av_find_best_stream(decoder->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (decoder->video_stream < 0) {
        set_av_error(errbuf, errbuf_size, "av_find_best_stream", decoder->video_stream);
        goto fail;
    }

    stream = decoder->fmt->streams[decoder->video_stream];
    codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        set_error(errbuf, errbuf_size, "decoder not found");
        goto fail;
    }

    if (create_primary_cuda_hwdevice(&decoder->hw_device_ctx, errbuf, errbuf_size) != 0) {
        goto fail;
    }

    decoder->ctx = avcodec_alloc_context3(codec);
    if (!decoder->ctx) {
        set_error(errbuf, errbuf_size, "avcodec_alloc_context3 failed");
        goto fail;
    }
    err = avcodec_parameters_to_context(decoder->ctx, stream->codecpar);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_parameters_to_context", err);
        goto fail;
    }
    decoder->ctx->get_format = get_hw_format;
    decoder->ctx->hw_device_ctx = av_buffer_ref(decoder->hw_device_ctx);

    err = avcodec_open2(decoder->ctx, codec, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_open2", err);
        goto fail;
    }

    decoder->packet = av_packet_alloc();
    decoder->frame = av_frame_alloc();
    if (!decoder->packet || !decoder->frame) {
        set_error(errbuf, errbuf_size, "packet/frame allocation failed");
        goto fail;
    }
    if (cudaStreamCreateWithFlags(&decoder->stream, cudaStreamNonBlocking) != cudaSuccess) {
        set_error(errbuf, errbuf_size, "cudaStreamCreateWithFlags bridge failed");
        goto fail;
    }

    return decoder;

fail:
    if (decoder) {
        av_frame_free(&decoder->frame);
        av_packet_free(&decoder->packet);
        avcodec_free_context(&decoder->ctx);
        av_buffer_unref(&decoder->hw_device_ctx);
        avformat_close_input(&decoder->fmt);
        if (decoder->stream) {
            cudaStreamDestroy(decoder->stream);
        }
        delete decoder;
    }
    return NULL;
}

extern "C"
int decode_next_p010_to_chw_fp16(
    BridgeDecoder *decoder,
    void *dst_device,
    char *errbuf,
    int errbuf_size
) {
    if (!decoder || !dst_device) {
        set_error(errbuf, errbuf_size, "decoder and dst_device are required");
        return -1;
    }

    while (1) {
        int err = avcodec_receive_frame(decoder->ctx, decoder->frame);
        if (err == 0) {
            int converted = convert_frame_to_chw_fp16(
                decoder->frame,
                dst_device,
                decoder->width,
                decoder->height,
                decoder->stream,
                errbuf,
                errbuf_size
            );
            av_frame_unref(decoder->frame);
            if (converted != 0) {
                return -1;
            }
            return 1;
        }
        if (err == AVERROR_EOF) {
            return 0;
        }
        if (err != AVERROR(EAGAIN)) {
            set_av_error(errbuf, errbuf_size, "avcodec_receive_frame", err);
            return -1;
        }

        if (decoder->sent_eof) {
            return 0;
        }

        while (1) {
            err = av_read_frame(decoder->fmt, decoder->packet);
            if (err == AVERROR_EOF) {
                avcodec_send_packet(decoder->ctx, NULL);
                decoder->sent_eof = 1;
                break;
            }
            if (err < 0) {
                set_av_error(errbuf, errbuf_size, "av_read_frame", err);
                return -1;
            }
            if (decoder->packet->stream_index != decoder->video_stream) {
                av_packet_unref(decoder->packet);
                continue;
            }

            err = avcodec_send_packet(decoder->ctx, decoder->packet);
            av_packet_unref(decoder->packet);
            if (err == AVERROR(EAGAIN)) {
                break;
            }
            if (err < 0) {
                set_av_error(errbuf, errbuf_size, "avcodec_send_packet", err);
                return -1;
            }
            break;
        }
    }
}

extern "C"
void close_p010_decoder(BridgeDecoder *decoder) {
    if (!decoder) {
        return;
    }
    av_frame_free(&decoder->frame);
    av_packet_free(&decoder->packet);
    avcodec_free_context(&decoder->ctx);
    av_buffer_unref(&decoder->hw_device_ctx);
    avformat_close_input(&decoder->fmt);
    if (decoder->stream) {
        cudaStreamDestroy(decoder->stream);
    }
    delete decoder;
}

extern "C"
BridgeNvencWriter *open_cuda_nvenc_writer(
    const char *output,
    int width,
    int height,
    int fps,
    const char *bitrate,
    int gop_size,
    char *errbuf,
    int errbuf_size
) {
    if (!output || width <= 0 || height <= 0 || fps <= 0) {
        set_error(errbuf, errbuf_size, "output, width, height, and fps are required");
        return NULL;
    }

    BridgeNvencWriter *writer = new BridgeNvencWriter();
    memset(writer, 0, sizeof(*writer));
    writer->stream_index = -1;
    writer->width = width;
    writer->height = height;
    writer->fps = fps;

    int err = 0;
    AVDictionary *mux_opts = NULL;
    AVStream *stream = NULL;
    const AVCodec *codec = NULL;

    if (create_primary_cuda_hwdevice(&writer->hw_device_ctx, errbuf, errbuf_size) != 0) {
        goto fail;
    }

    writer->frames_ctx = av_hwframe_ctx_alloc(writer->hw_device_ctx);
    if (!writer->frames_ctx) {
        set_error(errbuf, errbuf_size, "av_hwframe_ctx_alloc failed");
        goto fail;
    }
    {
        AVHWFramesContext *frames = (AVHWFramesContext *)writer->frames_ctx->data;
        frames->format = AV_PIX_FMT_CUDA;
        frames->sw_format = AV_PIX_FMT_NV12;
        frames->width = width;
        frames->height = height;
        frames->initial_pool_size = 8;
    }
    err = av_hwframe_ctx_init(writer->frames_ctx);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "av_hwframe_ctx_init cuda/nv12", err);
        goto fail;
    }

    err = avformat_alloc_output_context2(&writer->fmt, NULL, NULL, output);
    if (err < 0 || !writer->fmt) {
        set_av_error(errbuf, errbuf_size, "avformat_alloc_output_context2", err);
        goto fail;
    }

    codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) {
        set_error(errbuf, errbuf_size, "hevc_nvenc encoder not found");
        goto fail;
    }
    stream = avformat_new_stream(writer->fmt, NULL);
    if (!stream) {
        set_error(errbuf, errbuf_size, "avformat_new_stream failed");
        goto fail;
    }
    writer->stream_index = stream->index;

    writer->enc = avcodec_alloc_context3(codec);
    if (!writer->enc) {
        set_error(errbuf, errbuf_size, "avcodec_alloc_context3 failed");
        goto fail;
    }
    writer->enc->width = width;
    writer->enc->height = height;
    writer->enc->time_base = AVRational{1, fps};
    writer->enc->framerate = AVRational{fps, 1};
    writer->enc->pix_fmt = AV_PIX_FMT_CUDA;
    writer->enc->bit_rate = parse_bitrate(bitrate);
    writer->enc->gop_size = gop_size > 0 ? gop_size : fps * 2;
    writer->enc->max_b_frames = 0;
    writer->enc->hw_frames_ctx = av_buffer_ref(writer->frames_ctx);
    if (writer->fmt->oformat->flags & AVFMT_GLOBALHEADER) {
        writer->enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    {
        AVDictionary *opts = NULL;
        av_dict_set(&opts, "preset", "p1", 0);
        av_dict_set(&opts, "tune", "ull", 0);
        av_dict_set(&opts, "forced-idr", "1", 0);
        err = avcodec_open2(writer->enc, codec, &opts);
        av_dict_free(&opts);
    }
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_open2 hevc_nvenc", err);
        goto fail;
    }

    err = avcodec_parameters_from_context(stream->codecpar, writer->enc);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_parameters_from_context", err);
        goto fail;
    }
    stream->time_base = writer->enc->time_base;

    if (!(writer->fmt->oformat->flags & AVFMT_NOFILE)) {
        err = avio_open(&writer->fmt->pb, output, AVIO_FLAG_WRITE);
        if (err < 0) {
            set_av_error(errbuf, errbuf_size, "avio_open output", err);
            goto fail;
        }
    }

    av_dict_set(&mux_opts, "movflags", "+faststart", 0);
    err = avformat_write_header(writer->fmt, &mux_opts);
    av_dict_free(&mux_opts);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avformat_write_header", err);
        goto fail;
    }

    return writer;

fail:
    if (writer) {
        av_dict_free(&mux_opts);
        avcodec_free_context(&writer->enc);
        av_buffer_unref(&writer->frames_ctx);
        av_buffer_unref(&writer->hw_device_ctx);
        if (writer->fmt) {
            if (!(writer->fmt->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&writer->fmt->pb);
            }
            avformat_free_context(writer->fmt);
        }
        delete writer;
    }
    return NULL;
}

extern "C"
BridgeNvencFrame *begin_cuda_nvenc_frame(
    BridgeNvencWriter *writer,
    void **y_device,
    void **uv_device,
    int *y_pitch,
    int *uv_pitch,
    char *errbuf,
    int errbuf_size
) {
    if (!writer || !y_device || !uv_device || !y_pitch || !uv_pitch) {
        set_error(errbuf, errbuf_size, "writer and output pointers are required");
        return NULL;
    }
    BridgeNvencFrame *holder = new BridgeNvencFrame();
    memset(holder, 0, sizeof(*holder));
    holder->frame = av_frame_alloc();
    if (!holder->frame) {
        set_error(errbuf, errbuf_size, "av_frame_alloc failed");
        delete holder;
        return NULL;
    }
    int err = av_hwframe_get_buffer(writer->frames_ctx, holder->frame, 0);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "av_hwframe_get_buffer", err);
        av_frame_free(&holder->frame);
        delete holder;
        return NULL;
    }
    holder->frame->width = writer->width;
    holder->frame->height = writer->height;
    *y_device = holder->frame->data[0];
    *uv_device = holder->frame->data[1];
    *y_pitch = holder->frame->linesize[0];
    *uv_pitch = holder->frame->linesize[1];
    if (!*y_device || !*uv_device || *y_pitch <= 0 || *uv_pitch <= 0) {
        set_error(errbuf, errbuf_size, "invalid CUDA NV12 frame planes");
        av_frame_free(&holder->frame);
        delete holder;
        return NULL;
    }
    return holder;
}

extern "C"
int send_cuda_nvenc_frame(
    BridgeNvencWriter *writer,
    BridgeNvencFrame *holder,
    int64_t pts,
    char *errbuf,
    int errbuf_size
) {
    if (!writer || !holder || !holder->frame) {
        set_error(errbuf, errbuf_size, "writer and frame are required");
        return -1;
    }
    holder->frame->pts = pts;
    int err = avcodec_send_frame(writer->enc, holder->frame);
    av_frame_free(&holder->frame);
    delete holder;
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_send_frame", err);
        return -1;
    }
    return write_nvenc_packets(writer, errbuf, errbuf_size);
}

extern "C"
int close_cuda_nvenc_writer(BridgeNvencWriter *writer, char *errbuf, int errbuf_size) {
    if (!writer) {
        return 0;
    }
    int ret = 0;
    int err = avcodec_send_frame(writer->enc, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_send_frame flush", err);
        ret = -1;
    } else if (write_nvenc_packets(writer, errbuf, errbuf_size) != 0) {
        ret = -1;
    }
    if (ret == 0) {
        err = av_write_trailer(writer->fmt);
        if (err < 0) {
            set_av_error(errbuf, errbuf_size, "av_write_trailer", err);
            ret = -1;
        }
    }
    avcodec_free_context(&writer->enc);
    av_buffer_unref(&writer->frames_ctx);
    av_buffer_unref(&writer->hw_device_ctx);
    if (writer->fmt) {
        if (!(writer->fmt->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&writer->fmt->pb);
        }
        avformat_free_context(writer->fmt);
    }
    delete writer;
    return ret;
}

extern "C"
int decode_first_p010_to_chw_fp16(
    const char *input,
    void *dst_device,
    int expected_width,
    int expected_height,
    char *errbuf,
    int errbuf_size
) {
    if (!input || !dst_device) {
        set_error(errbuf, errbuf_size, "input and dst_device are required");
        return -1;
    }

    AVFormatContext *fmt = NULL;
    AVCodecContext *ctx = NULL;
    AVBufferRef *hw_device_ctx = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    int video_stream = -1;
    AVStream *stream = NULL;
    const AVCodec *codec = NULL;
    int ret = -1;

    int err = avformat_open_input(&fmt, input, NULL, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avformat_open_input", err);
        goto done;
    }
    err = avformat_find_stream_info(fmt, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avformat_find_stream_info", err);
        goto done;
    }

    video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_stream < 0) {
        set_av_error(errbuf, errbuf_size, "av_find_best_stream", video_stream);
        goto done;
    }

    stream = fmt->streams[video_stream];
    codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        set_error(errbuf, errbuf_size, "decoder not found");
        goto done;
    }

    if (create_primary_cuda_hwdevice(&hw_device_ctx, errbuf, errbuf_size) != 0) {
        goto done;
    }

    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        set_error(errbuf, errbuf_size, "avcodec_alloc_context3 failed");
        goto done;
    }
    err = avcodec_parameters_to_context(ctx, stream->codecpar);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_parameters_to_context", err);
        goto done;
    }
    ctx->get_format = get_hw_format;
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    err = avcodec_open2(ctx, codec, NULL);
    if (err < 0) {
        set_av_error(errbuf, errbuf_size, "avcodec_open2", err);
        goto done;
    }

    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (!packet || !frame) {
        set_error(errbuf, errbuf_size, "packet/frame allocation failed");
        goto done;
    }

    while ((err = av_read_frame(fmt, packet)) >= 0) {
        if (packet->stream_index != video_stream) {
            av_packet_unref(packet);
            continue;
        }

        err = avcodec_send_packet(ctx, packet);
        av_packet_unref(packet);
        if (err < 0) {
            set_av_error(errbuf, errbuf_size, "avcodec_send_packet", err);
            goto done;
        }

        err = avcodec_receive_frame(ctx, frame);
        if (err == AVERROR(EAGAIN)) {
            continue;
        }
        if (err < 0) {
            set_av_error(errbuf, errbuf_size, "avcodec_receive_frame", err);
            goto done;
        }

        cudaStream_t stream = NULL;
        cudaError_t stream_err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (stream_err != cudaSuccess) {
            set_error(errbuf, errbuf_size, "cudaStreamCreateWithFlags bridge failed: %s", cudaGetErrorString(stream_err));
            goto done;
        }
        if (convert_frame_to_chw_fp16(frame, dst_device, expected_width, expected_height, stream, errbuf, errbuf_size) != 0) {
            cudaStreamDestroy(stream);
            goto done;
        }
        cudaStreamDestroy(stream);

        ret = 1;
        goto done;
    }

    if (err == AVERROR_EOF) {
        set_error(errbuf, errbuf_size, "no video frame decoded");
    } else {
        set_av_error(errbuf, errbuf_size, "av_read_frame", err);
    }

done:
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&ctx);
    av_buffer_unref(&hw_device_ctx);
    avformat_close_input(&fmt);
    return ret;
}
