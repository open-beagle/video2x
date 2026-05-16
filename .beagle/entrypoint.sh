#!/usr/bin/env bash
set -euo pipefail

if [[ "${VIDEO_ENCODER:-}" == "hevc_nvenc" || "${VIDEO_ENCODER:-}" == "h264_nvenc" ]]; then
  if [[ ! -f /opt/hooks/nvenc_ioctl_hook.so ]]; then
    echo "ERROR: NVENC hook missing: /opt/hooks/nvenc_ioctl_hook.so" >&2
    exit 1
  fi

  if [[ "${LD_PRELOAD:-}" != *"/opt/hooks/nvenc_ioctl_hook.so"* ]]; then
    echo "ERROR: LD_PRELOAD must include /opt/hooks/nvenc_ioctl_hook.so for NVENC" >&2
    exit 1
  fi

  if [[ -z "${NVENC_GPU_INDEX:-}" ]]; then
    echo "ERROR: NVENC_GPU_INDEX is required for NVENC hook" >&2
    exit 1
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found" >&2
    exit 1
  fi

  if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "${VIDEO_ENCODER}"; then
    echo "ERROR: ffmpeg encoder not available: ${VIDEO_ENCODER}" >&2
    exit 1
  fi
fi

exec python3 /app/src/cli.py "$@"
