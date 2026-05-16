#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAME="${MODEL_NAME:-realesr-general-x4v3}"
WEIGHTS="${MODEL_DIR}/${MODEL_NAME}.pth"

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

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: weights not found: ${WEIGHTS}" >&2
  exit 1
fi

echo "MODEL_DIR=${MODEL_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "WEIGHTS=${WEIGHTS}"

for profile in "${PROFILES[@]}"; do
  width="${profile%x*}"
  height="${profile#*x}"
  onnx="${MODEL_DIR}/${MODEL_NAME}-${width}x${height}.onnx"
  engine="${MODEL_DIR}/${MODEL_NAME}-${width}x${height}-fp16.engine"

  echo
  echo "== ${MODEL_NAME} ${width}x${height} =="

  if [[ ! -f "${onnx}" ]]; then
    python /app/tools/export_realesrgan_onnx.py \
      --model "${MODEL_NAME}" \
      --weights "${WEIGHTS}" \
      --output "${onnx}" \
      --input-shape "1,3,${height},${width}"
  else
    echo "ONNX exists: ${onnx}"
  fi

  if [[ ! -f "${engine}" ]]; then
    build_engine "${onnx}" "${engine}"
  else
    echo "engine exists: ${engine}"
  fi
done

echo
echo "Done. Engine files:"
find "${MODEL_DIR}" -maxdepth 1 -type f -name "${MODEL_NAME}-*-fp16.engine" -printf "%f\n" | sort
