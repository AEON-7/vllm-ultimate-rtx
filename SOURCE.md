# SOURCE — AEON vLLM Ultimate RTX

**RTX 5090 · Blackwell sm_120 · NVFP4**

This image is **not** for DGX Spark / GB10 / sm_121a. That provenance lives in
[AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark).

## Arch (do not invent, do not leave native)

| Token | Value | Who said so |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 (GB202) | `nvidia-smi` on the Aorus WSL Ubuntu-24.04 host |
| Compute capability | 12.0 | NVIDIA / PyTorch (`CUDA capability sm_120`) |
| `sm_*` | `sm_120` | CUDA 13.2 `nvcc --help` lists `sm_120` |
| `sm_120a` | accepted | same `nvcc --help` lists `sm_120a` / `compute_120a` / `compute_120` |
| `TORCH_CUDA_ARCH_LIST` | `12.0;12.0a` | 12.0 = 5090 identity. 12.0a = NVFP4 'a' family (Spark needed `12.1a`, not bare `12.1`) |
| `CMAKE_CUDA_ARCHITECTURES` | `120;120a` | same |
| Not compiled | `12.1` / `12.1a` (GB10), `8.9` (Ada), `sm_100` (B200), `native` | a 5090 file that claims these as the target is a bug |

Verified 2026-08-21 on spark-ecdd's CUDA 13.2 nvcc (`V13.2.86`): allowed values include
`compute_120`, `compute_120a`, `compute_120f`, `sm_120`, `sm_120a`, `sm_120f`.
We compile `120` and `120a` only. `120f` is not required for this bake.

## vLLM tree pin

Same AEON 0.27.1 3-way merge as the Spark image. Not vendored (~140 MB).

| | |
|---|---|
| Branch | `aeon-v0.27.1` |
| SHA | `34aeb2303114bf3d8b2ecd582b7301868691735a` |
| Message | `[Model Runner V2] Cache draft logits in model's LM head dtype (#50910)` |
| On spark-ecdd | `/home/albert/aeon-builds/vllm-ultimate-v0.27.1/vllm-src` |
| Remotes | `origin` lesj0610/vllm (historical NVFP4-KV fork), `upstream` vllm-project/vllm |
| Upstream tag underneath | v0.27.1 (`6e448d0ea9` = v0.27.0 + #50424) |

Carries that still apply on sm_120 (kept):

- #44389 Triton NVFP4-KV, NHD-safe slicing (upstream `as_strided` still unsafe)
- NVFP4_AWQ deserializer
- #41703 ctx-slot masking, DFlash SWA / block-table unpad
- #50276 packed-KV zeroer stride
- #51812 GDN gate / spec-token alignment
- #51843 hybrid-APC
- #48053 thread_local capture
- #46932 UMA clamp (harmless on dedicated VRAM; not a 5090 default path)

Not baked as *defaults* on this image (present in the tree, unused unless you opt in):

- DSpark Markov heads / top-k projection
- `VLLM_USE_V2_MODEL_RUNNER` pin
- GB10 `TRITON_ATTN` serve recipes
- MTP=2 as CMD

## Dep pins (mirror Spark 0.27.1, x86_64 wheels)

| Package | Version |
|---|---|
| torch | 2.13.0+cu130 |
| torchvision | 0.28.0 |
| triton | 3.7.1 (pulled by torch) |
| flashinfer-python / cubin / jit-cache | 0.6.16.post3 |
| nvidia-cutlass-dsl | 4.6.0 |
| quack-kernels | 0.6.1 exact |
| apache-tvm-ffi | 0.1.11 |
| tilelang | 0.1.12 |
| nvidia-nccl-cu13 | 2.30.7 |
| transformers | 5.14.1 |
| torchcodec | 0.16.0 |
| xgrammar | >=0.2.1,<1.0.0 |
| turboquant | AEON-7 `fix/cuda-graph-safe-qjl-powers` |
| CUDA toolkit (image) | 13.0.2 devel ubuntu22.04, linux/amd64 |

## Bake command

```bash
# linux/amd64 only. Script refuses aarch64 (spark-ecdd).
# Do not start Docker Desktop on Aorus.
IMAGE=ghcr.io/aeon-7/aeon-vllm-ultimate-rtx \
TAG=sm120 \
VLLM_SRC=/home/albert/aeon-builds/vllm-ultimate-v0.27.1/vllm-src \
  ./scripts/bake-rtx5090.sh
```

Preferred bake hosts: GitHub-hosted/large `linux/amd64` runner, or a native
x86_64 docker engine that is **already running**. Not Spark. Not Docker Desktop
on the Aorus (pgembed `:5432`).

## Image names

| | |
|---|---|
| Registry | `ghcr.io/aeon-7/aeon-vllm-ultimate-rtx` |
| Tags | `:sm120` `:5090` `:rtx5090` |
| Do not touch | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (Spark) |
