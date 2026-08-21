#!/usr/bin/env bash
# Fail if any 5090-target file claims sm_121 / 12.1a as THE compile target,
# or leaves TORCH_CUDA_ARCH_LIST unset/native, or sneaks in Ada 8.9 as default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail=0
# Files that MUST pin 12.0 / sm_120 as the 5090 target.
files=(Dockerfile Dockerfile.slim docker-compose.judge.yml README.md SOURCE.md AGENTS.md
       scripts/bake-rtx5090.sh .github/workflows/ci.yml .github/workflows/image.yml)

for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "MISSING $f"; fail=1; continue; }
done

# Dockerfile must set the exact pin (12.0 present; 12.0a allowed).
if ! grep -q 'TORCH_CUDA_ARCH_LIST="12.0;12.0a"' Dockerfile; then
  echo "Dockerfile missing TORCH_CUDA_ARCH_LIST=12.0;12.0a"
  fail=1
fi
if ! grep -q 'CMAKE_CUDA_ARCHITECTURES="120;120a"' Dockerfile; then
  echo "Dockerfile missing CMAKE_CUDA_ARCHITECTURES=120;120a"
  fail=1
fi
if grep -E 'ENV TORCH_CUDA_ARCH_LIST="(native|12\.1a?|8\.9)"' Dockerfile Dockerfile.slim; then
  echo "bad ENV TORCH_CUDA_ARCH_LIST"
  fail=1
fi
if grep -E 'ENV VLLM_USE_V2_MODEL_RUNNER=' Dockerfile Dockerfile.slim; then
  echo "V2 runner must not be baked on the 5090 path"
  fail=1
fi

# 5090 identity strings must appear.
for f in Dockerfile README.md SOURCE.md; do
  if ! grep -q 'sm_120' "$f"; then
    echo "$f missing sm_120"
    fail=1
  fi
  if ! grep -q '12.0' "$f"; then
    echo "$f missing 12.0"
    fail=1
  fi
done

# Spark comparison is allowed; claiming 12.1a IS the 5090 target is not.
if grep -nE 'TORCH_CUDA_ARCH_LIST=12\.1a' Dockerfile Dockerfile.slim scripts/bake-rtx5090.sh .github/workflows/image.yml; then
  echo "12.1a used as a 5090 compile pin — bug"
  fail=1
fi

if [[ $fail -ne 0 ]]; then
  echo "arch-pin check FAILED"
  exit 1
fi
echo "arch-pin check OK — RTX 5090 · Blackwell sm_120 · NVFP4"
