#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
TARGET_HEIGHT="${TARGET_HEIGHT:-1080}"
MODEL_NAME="${MODEL_NAME:-RealESRGAN_x2plus}"
OUTSCALE="${OUTSCALE:-}"
TILE="${TILE:-0}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_1080p}"
BENCHMARK_FRAMES="${BENCHMARK_FRAMES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
REALESRGAN_HOME="${REALESRGAN_HOME:-/opt/Real-ESRGAN}"

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

probe_field() {
  local file="$1"
  local field="$2"
  ffprobe -v error -select_streams v:0 -show_entries "stream=${field}" \
    -of default=noprint_wrappers=1:nokey=1 "$file" | head -n 1
}

probe_frames() {
  local file="$1"
  local frames
  frames="$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames \
    -of default=noprint_wrappers=1:nokey=1 "$file" | head -n 1 || true)"
  if [[ -z "${frames}" || "${frames}" == "N/A" ]]; then
    frames="$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=nb_frames \
      -of default=noprint_wrappers=1:nokey=1 "$file" | head -n 1 || true)"
  fi
  printf '%s' "${frames:-unknown}"
}

calc_outscale() {
  local height="$1"
  python - "$TARGET_HEIGHT" "$height" <<'PY'
import sys
target = float(sys.argv[1])
height = float(sys.argv[2])
print(f"{target / height:.6f}".rstrip("0").rstrip("."))
PY
}

make_output_path() {
  local file="$1"
  local dir base name
  dir="$(dirname "$file")"
  base="$(basename "$file")"
  name="${base%.*}"
  printf '%s/%s%s.mp4' "$dir" "$name" "$OUTPUT_SUFFIX"
}

run_realesrgan() {
  local input="$1"
  local output="$2"
  local scale="$3"
  local work_input="$input"
  local tmpdir outdir produced suffix_name input_name

  tmpdir="$(mktemp -d)"
  outdir="${tmpdir}/out"
  mkdir -p "$outdir"
  suffix_name="${OUTPUT_SUFFIX#_}"

  if [[ -n "${BENCHMARK_FRAMES}" ]]; then
    work_input="${tmpdir}/benchmark.mp4"
    ffmpeg -hide_banner -loglevel error -y \
      -i "$input" -map 0:v:0 -an -frames:v "$BENCHMARK_FRAMES" "$work_input"
  fi

  local start end elapsed
  start="$(date +%s)"
  (
    cd "$REALESRGAN_HOME"
    python inference_realesrgan_video.py \
      -i "$work_input" \
      -o "$outdir" \
      -n "$MODEL_NAME" \
      -s "$scale" \
      --suffix "$suffix_name" \
      --tile "$TILE"
  )
  end="$(date +%s)"
  elapsed="$((end - start))"

  input_name="$(basename "${work_input%.*}")"
  produced="${outdir}/${input_name}_${suffix_name}.mp4"
  [[ -s "$produced" ]] || die "Real-ESRGAN output not found: $produced"
  mv -f "$produced" "$output"

  if [[ -n "${BENCHMARK_FRAMES}" ]]; then
    log "Benchmark elapsed: ${elapsed}s for ${BENCHMARK_FRAMES} frames"
  fi

  rm -rf "$tmpdir"
}

validate_output() {
  local file="$1"
  [[ -s "$file" ]] || die "output missing or empty: $file"

  local height
  height="$(probe_field "$file" height || true)"
  [[ "$height" == "$TARGET_HEIGHT" ]] || die "output height ${height:-unknown} != ${TARGET_HEIGHT}: $file"
}

require_command ffmpeg
require_command ffprobe
require_command python

[[ -d "$DATA_DIR" ]] || die "DATA_DIR does not exist: $DATA_DIR"
[[ -f "${REALESRGAN_HOME}/weights/${MODEL_NAME}.pth" ]] || die "model not found: ${REALESRGAN_HOME}/weights/${MODEL_NAME}.pth"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

python - <<'PY' || die "CUDA is not available to PyTorch"
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY

log "DATA_DIR=${DATA_DIR}"
log "TARGET_HEIGHT=${TARGET_HEIGHT}"
log "MODEL_NAME=${MODEL_NAME}"
log "GPU_ID=${GPU_ID}"
log "TILE=${TILE}"
log "OUTPUT_SUFFIX=${OUTPUT_SUFFIX}"
log "SKIP_EXISTING=${SKIP_EXISTING}"
[[ -n "$BENCHMARK_FRAMES" ]] && log "BENCHMARK_FRAMES=${BENCHMARK_FRAMES}"

mapfile -d '' inputs < <(find "$DATA_DIR" -type f -iname '*.mp4' ! -iname "*${OUTPUT_SUFFIX}.mp4" -print0 | sort -z)
[[ "${#inputs[@]}" -gt 0 ]] || {
  log "No input .mp4 files found."
  exit 0
}

task_no=0
for input in "${inputs[@]}"; do
  output="$(make_output_path "$input")"
  if [[ -n "$BENCHMARK_FRAMES" ]]; then
    output="${output%.mp4}_benchmark.mp4"
  fi

  if [[ "$SKIP_EXISTING" == "true" && -e "$output" ]]; then
    log "Skip existing: $output"
    continue
  fi

  width="$(probe_field "$input" width || true)"
  height="$(probe_field "$input" height || true)"
  frames="$(probe_frames "$input")"

  if [[ -z "$height" || -z "$width" ]]; then
    log "Skip unreadable video: $input"
    continue
  fi

  if (( height >= TARGET_HEIGHT )); then
    log "Skip ${width}x${height}: $input"
    continue
  fi

  scale="${OUTSCALE:-$(calc_outscale "$height")}"
  task_no="$((task_no + 1))"

  log ""
  log "${task_no}. ${input}"
  log "   input: ${width}x${height}, ${frames} frames"
  log "   output: ${output}"
  log "   model: ${MODEL_NAME}"
  log "   outscale: ${scale}"

  run_realesrgan "$input" "$output" "$scale"
  validate_output "$output"
done

log ""
log "Done. processed tasks: ${task_no}"
