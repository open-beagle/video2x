#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 INPUT OUTPUT [FRAMES]" >&2
  exit 2
fi

input="$1"
output="$2"
frames="${3:-300}"

echo "== ffmpeg hwaccels =="
ffmpeg -hide_banner -hwaccels | sed -n '1,80p'

echo "== ffmpeg cuda filters =="
ffmpeg -hide_banner -filters | grep -E '(cuda|npp|hwupload|hwdownload|scale_)' || true

echo "== ffmpeg nvenc encoders =="
ffmpeg -hide_banner -encoders | grep nvenc || true

echo "== ffmpeg cuvid decoders =="
ffmpeg -hide_banner -decoders | grep cuvid || true

echo "== cuda surface smoke test =="
ffmpeg -hide_banner -loglevel info -y \
  -hwaccel cuda \
  -hwaccel_output_format cuda \
  -i "${input}" \
  -vf scale_cuda=1920:1080:format=nv12 \
  -an \
  -c:v hevc_nvenc \
  -preset p1 \
  -tune ull \
  -b:v 5M \
  -frames:v "${frames}" \
  "${output}"

echo "== output =="
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  "${output}"
