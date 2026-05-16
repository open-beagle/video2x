#!/usr/bin/env bash
set -euo pipefail

detect_nvidia_gpu_index() {
  local dev
  for dev in /dev/nvidia[0-9]*; do
    [[ -e "${dev}" ]] || continue
    [[ "${dev}" =~ ^/dev/nvidia([0-9]+)$ ]] || continue
    echo "${BASH_REMATCH[1]}"
    return 0
  done
  return 1
}

prepare_nvenc_cdi_device_alias() {
  local target_index="$1"
  local target="/dev/nvidia${target_index}"

  [[ -e "${target}" ]] || return 0

  if [[ "${target_index}" != "0" && ! -e /dev/nvidia0 ]]; then
    ln -s "${target}" /dev/nvidia0
  fi
}

if [[ "${VIDEO_ENCODER:-}" == "hevc_nvenc" || "${VIDEO_ENCODER:-}" == "h264_nvenc" ]]; then
  if [[ ! -f /opt/hooks/nvenc_ioctl_hook.so ]]; then
    echo "ERROR: NVENC hook missing: /opt/hooks/nvenc_ioctl_hook.so" >&2
    exit 1
  fi

  if [[ -z "${NVENC_GPU_INDEX:-}" ]]; then
    detected_gpu_index="$(detect_nvidia_gpu_index || true)"
    if [[ -z "${detected_gpu_index}" ]]; then
      echo "ERROR: no /dev/nvidia<N> device found for NVENC hook" >&2
      exit 1
    fi
    prepare_nvenc_cdi_device_alias "${detected_gpu_index}"
  fi

  if [[ "${LD_PRELOAD:-}" != *"/opt/hooks/nvenc_ioctl_hook.so"* ]]; then
    echo "ERROR: LD_PRELOAD must include /opt/hooks/nvenc_ioctl_hook.so for NVENC" >&2
    exit 1
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found" >&2
    exit 1
  fi

  encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
  if ! grep -q "${VIDEO_ENCODER}" <<< "${encoders}"; then
    echo "ERROR: ffmpeg encoder not available: ${VIDEO_ENCODER}" >&2
    exit 1
  fi
fi

exec python3 /app/src/cli.py "$@"
