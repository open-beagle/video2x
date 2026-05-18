#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"

SUPPORTED_MODELS=(
  "realesr-general-x4v3"
  "realesr-general-wdn-x4v3"
  "RealESRGAN_x2plus"
  "RealESRGAN_x4plus"
)

PROFILES=(
  "640x360"
  "720x480"
  "854x480"
  "960x540"
  "1280x720"
  "720x420"
)

build_engine() {
  local onnx_path="$1"
  local engine_path="$2"

  if [[ ! -f "${onnx_path}" ]]; then
    echo "ERROR: ONNX file not found: ${onnx_path}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${engine_path}")"

  trtexec \
    --onnx="${onnx_path}" \
    --saveEngine="${engine_path}" \
    --fp16
}

build_conv48_engine() {
  local model="$1"
  local width="$2"
  local height="$3"
  local onnx="$4"
  local conv48_onnx="${MODEL_DIR}/${model}-${width}x${height}-conv48.onnx"
  local conv48_engine="${MODEL_DIR}/${model}-${width}x${height}-conv48-fp16.engine"
  local conv48_tail="${MODEL_DIR}/${model}-tail-${width}x${height}-conv48.npz"

  if [[ "${model}" != "realesr-general-x4v3" && "${model}" != "realesr-general-wdn-x4v3" ]]; then
    return 0
  fi

  if [[ ! -f "${conv48_onnx}" ]]; then
    python /app/tools/split_srvgg_tail_onnx.py \
      --input "${onnx}" \
      --output "${conv48_onnx}" \
      --tail-weights "${conv48_tail}" \
      --split-at post-conv
  else
    echo "conv48 ONNX exists: ${conv48_onnx}"
  fi

  if [[ ! -f "${conv48_engine}" ]]; then
    build_engine "${conv48_onnx}" "${conv48_engine}"
  else
    echo "conv48 engine exists: ${conv48_engine}"
  fi
}

model_supported() {
  local name="$1"
  local supported
  for supported in "${SUPPORTED_MODELS[@]}"; do
    if [[ "${supported}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "MODEL_DIR=${MODEL_DIR}"

MODELS=()
if [[ -n "${MODEL_NAME:-}" && "${MODEL_NAME}" != "all" ]]; then
  if ! model_supported "${MODEL_NAME}"; then
    echo "ERROR: unsupported MODEL_NAME=${MODEL_NAME}" >&2
    echo "supported: ${SUPPORTED_MODELS[*]}" >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_DIR}/${MODEL_NAME}.pth" ]]; then
    echo "ERROR: weights not found: ${MODEL_DIR}/${MODEL_NAME}.pth" >&2
    exit 1
  fi
  MODELS+=("${MODEL_NAME}")
else
  for model in "${SUPPORTED_MODELS[@]}"; do
    if [[ -f "${MODEL_DIR}/${model}.pth" ]]; then
      MODELS+=("${model}")
    else
      echo "skip missing weights: ${MODEL_DIR}/${model}.pth"
    fi
  done
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "ERROR: no supported .pth weights found in ${MODEL_DIR}" >&2
  echo "supported: ${SUPPORTED_MODELS[*]}" >&2
  exit 1
fi

echo "MODELS=${MODELS[*]}"

for model in "${MODELS[@]}"; do
  weights="${MODEL_DIR}/${model}.pth"
  echo
  echo "## model=${model}"
  echo "weights=${weights}"

  for profile in "${PROFILES[@]}"; do
    width="${profile%x*}"
    height="${profile#*x}"
    onnx="${MODEL_DIR}/${model}-${width}x${height}.onnx"
    engine="${MODEL_DIR}/${model}-${width}x${height}-fp16.engine"

    echo
    echo "== ${model} ${width}x${height} =="

    if [[ ! -f "${onnx}" ]]; then
      python /app/tools/export_realesrgan_onnx.py \
        --model "${model}" \
        --weights "${weights}" \
        --output "${onnx}" \
        --input-shape "1,3,${height},${width}" \
        --fp16
    else
      echo "ONNX exists: ${onnx}"
    fi

    if [[ ! -f "${engine}" ]]; then
      build_engine "${onnx}" "${engine}"
    else
      echo "engine exists: ${engine}"
    fi

    build_conv48_engine "${model}" "${width}" "${height}" "${onnx}"
  done
done

echo
echo "Done. Engine files:"
find "${MODEL_DIR}" -maxdepth 1 -type f -name "*-fp16.engine" -printf "%f\n" | sort
