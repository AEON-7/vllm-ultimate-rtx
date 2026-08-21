# AGENTS — vllm-ultimate-rtx

You are on the **RTX 5090 / sm_120** lane. If a file says `sm_121` or `12.1a`
as the 5090 target, that is a bug. Fix it. Do not "also compile Spark, one image
for the fleet" — that preference is retired. The split is real.

## Identity

- Repo: `AEON-7/vllm-ultimate-rtx` (public)
- Image: `ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:{sm120,5090,rtx5090}`
- Arch: `TORCH_CUDA_ARCH_LIST=12.0;12.0a` · `CMAKE_CUDA_ARCHITECTURES=120;120a`
- Hardware: GeForce RTX 5090 (GB202), dedicated 32 GB, WSL Ubuntu-24.04
- Sibling: `AEON-7/vllm-ultimate-dgx-spark` / `sm_121a` / GB10 — not this image

## Never

- Start Docker Desktop on Aorus (pgembed `:5432`)
- Start a Qwen3.8-27B hunt on the 5090 as part of a bake
- Bounce friend Brev GPUs (0942/00c9)
- Print secrets / tokens
- Hub-push models
- `git clean -x` / force-push
- Bake DSpark / `VLLM_USE_V2_MODEL_RUNNER` / GB10 `TRITON_ATTN` as defaults
- Set `TORCH_CUDA_ARCH_LIST` to `native`, unset, `8.9`, `12.1`, or `12.1a`
- Merge a PR unless CI is green **and** the change is clearly the 5090 lane
- Clone AEON-7 work onto the assistant box when spark-ecdd `gh` works

## Serve seats

**Judge (default documented):** port `18080`, `--max-model-len 8192`,
`--max-num-seqs 64`, `--kv-cache-dtype nvfp4`, `--attention-backend FLASHINFER`,
thinking off at the request, **no MTP**.

**Long-run chat/student (later):** longer ctx, optional DFlash, optional MTP=2.
Not the image CMD. Not the judge seat.

## Bake

`scripts/bake-rtx5090.sh` refuses aarch64. vllm-src pin is in SOURCE.md.
`scripts/check-arch-pins.sh` is the CI gate — keep it mean.

## Voice

AEON FORGE / @SpaceTimeViking. Punchy commits. Conceptual ethos + real flags.
No corporate how-to. No secret dumps.
