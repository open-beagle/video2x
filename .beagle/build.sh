#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-video2x}"
IMAGE_TAG="${IMAGE_TAG:-0.3.0}"
BASE="${BASE:-nvidia/cuda:13.0.3-runtime-ubuntu24.04}"
DOCKERFILE="${DOCKERFILE:-.beagle/dockerfile}"
CONTEXT="${CONTEXT:-.}"

docker build \
  --build-arg "BASE=${BASE}" \
  --build-arg "VERSION=${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${CONTEXT}"
