#!/usr/bin/env bash
set -euo pipefail

for t in 1 2 3 4; do
  ffmpeg -hide_banner -loglevel error -y -ss "$t" -i /data/SDMT-506-U-420p.mp4 -frames:v 1 "/data/compare/original_${t}s.jpg"
  ffmpeg -hide_banner -loglevel error -y -ss "$t" -i /data/SDMT-506-U-420p_1080p_general_x4v3.mp4 -frames:v 1 "/data/compare/pytorch_${t}s.jpg"
  ffmpeg -hide_banner -loglevel error -y -ss "$t" -i /data/SDMT-506-U-420p_trt_cuda_full.mp4 -frames:v 1 "/data/compare/trt_cuda_${t}s.jpg"
  ffmpeg -hide_banner -loglevel error -y \
    -i "/data/compare/original_${t}s.jpg" \
    -i "/data/compare/pytorch_${t}s.jpg" \
    -i "/data/compare/trt_cuda_${t}s.jpg" \
    -filter_complex "[0:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,drawtext=text='Original':fontcolor=white:fontsize=24:x=20:y=20:box=1:boxcolor=black@0.5[o];[1:v]scale=640:360,drawtext=text='PyTorch':fontcolor=white:fontsize=24:x=20:y=20:box=1:boxcolor=black@0.5[p];[2:v]scale=640:360,drawtext=text='TRT-CUDA':fontcolor=white:fontsize=24:x=20:y=20:box=1:boxcolor=black@0.5[t];[o][p][t]hstack=inputs=3" \
    "/data/compare/compare_${t}s.jpg"
done

ls -lh /data/compare
