#!/usr/bin/env bash
# Bake ghcr.io/aeon-7/aeon-vllm-ultimate-rtx for RTX 5090 / sm_120.
# Run on linux/amd64 with Docker already running.
# Do NOT run on DGX Spark (aarch64). Do NOT start Docker Desktop on Aorus.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate-rtx}"
TAG="${TAG:-sm120}"
VLLM_SRC="${VLLM_SRC:-$ROOT/vllm-src}"

if [[ ! -d "$VLLM_SRC" ]]; then
  echo "missing vllm-src at $VLLM_SRC" >&2
  echo "On spark-ecdd: VLLM_SRC=/home/albert/aeon-builds/vllm-ultimate-v0.27.1/vllm-src" >&2
  echo "Pin: aeon-v0.27.1 @ 34aeb2303114bf3d8b2ecd582b7301868691735a" >&2
  exit 1
fi

# Refuse to bake on aarch64 — that would be the Spark box compiling the wrong ISA.
arch="$(uname -m)"
if [[ "$arch" != "x86_64" && "$arch" != "amd64" ]]; then
  echo "refusing bake on $arch — RTX 5090 image is linux/amd64 / sm_120" >&2
  echo "Bake on GHCR Actions (linux/amd64) or a native x86_64 docker already running." >&2
  exit 1
fi

if [[ ! -d "$ROOT/vllm-src" ]]; then
  ln -s "$VLLM_SRC" "$ROOT/vllm-src"
fi

echo "Baking $IMAGE:$TAG"
echo "  TORCH_CUDA_ARCH_LIST=12.0;12.0a"
echo "  CMAKE_CUDA_ARCHITECTURES=120;120a"
echo "  vllm-src=$VLLM_SRC"

docker build \
  --platform linux/amd64 \
  -t "$IMAGE:$TAG" \
  -t "$IMAGE:5090" \
  -t "$IMAGE:rtx5090" \
  -f "$ROOT/Dockerfile" \
  "$ROOT"

echo "baked $IMAGE:$TAG  (also :5090 :rtx5090)"
echo "push: docker push $IMAGE:$TAG && docker push $IMAGE:5090 && docker push $IMAGE:rtx5090"
