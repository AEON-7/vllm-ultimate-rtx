# AEON vLLM Ultimate RTX

[![docker](https://img.shields.io/badge/ghcr.io-aeon--7%2Faeon--vllm--ultimate--rtx-blue?logo=docker)](https://github.com/users/AEON-7/packages/container/package/aeon-vllm-ultimate-rtx)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1%2Baeon.sm120.rtx-orange)](https://github.com/vllm-project/vllm/releases/tag/v0.27.1)
[![arch](https://img.shields.io/badge/linux%2Famd64-CUDA%2013%20sm__120-green)](https://github.com/AEON-7/vllm-ultimate-rtx)

**Public register:** AEON FORGE / [@SpaceTimeViking](https://github.com/SpaceTimeViking)

Slim, linux/amd64 vLLM image for **NVIDIA Blackwell consumer / workstation `sm_120`**: GeForce **RTX 5090** and **RTX PRO 6000 Blackwell**. Compiled for CUDA 13 (`12.0` + `12.0a`) so NVFP4 / FP8 kernels actually land. Vision + audio extras are in this public-tag slim.

GHCR package **`ghcr.io/aeon-7/aeon-vllm-ultimate-rtx`** is **private** (same visibility as `mf-*-rtx`). The GitHub repo is public so this card can live here. Pull requires a GHCR login as someone who can read the package.

**This tag is not Spark. This tag is not B200 / B300.**

| | This slim `:sm120` | Spark Ultimate (sibling) | Later / fat (not this tag) |
|---|---|---|---|
| **Repo** | [AEON-7/vllm-ultimate-rtx](https://github.com/AEON-7/vllm-ultimate-rtx) | [AEON-7/vllm-ultimate](https://github.com/AEON-7/vllm-ultimate) / [vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark) | same RTX repo |
| **Image** | `ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120` | `ghcr.io/aeon-7/aeon-vllm-ultimate` (aarch64) — **do not overwrite** | `:sm120-fat`, planned `:sm100` / `:sm103` |
| **ISA** | linux/amd64 | linux/aarch64 | linux/amd64 |
| **CUDA arch** | **sm_120** · `TORCH_CUDA_ARCH_LIST=12.0;12.0a` | sm_121a / GB10 / UMA | sm_100 / sm_103 cubins (stripped from this slim) |
| **Target GPUs** | RTX 5090, RTX PRO 6000 Blackwell | DGX Spark GB10 | B200 / B300 — **not claimed here** |
| **Memory model** | dedicated VRAM | UMA | dedicated |
| **Size (local)** | ~25.2 GB slim | Spark package is separate | ~49.8 GB `:sm120-fat` |

OCI title: `AEON vLLM Ultimate RTX slim (5090 / sm_120)`. Image version label: `0.27.1+aeon.sm120.rtx`. Labels also set `ai.aeon.not_for=DGX Spark GB10 sm_121a, B200 sm_100, B300 sm_103`.

---

## Pull

```bash
# WSL Ubuntu-24.04 + docker.io (not Docker Desktop)
echo "$GHCR_TOKEN" | docker login ghcr.io -u AEON-7 --password-stdin
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120
# same digest:
#   :aeon-vllm-rtx-sm120
# local aliases (same image id): :5090 :rtx5090
```

Published slim digest (this card):

`ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120@sha256:6f580a1c2f83aac1dafba88b59859d8e95bee0edde57144d935650b8e7e00e67`

Do **not** retag or push onto `ghcr.io/aeon-7/aeon-vllm-ultimate` (Spark / aarch64).

---

## What it is

- **FROM / base:** CUDA **13.0.2** devel stack (`linux/amd64`). `CUDA_VERSION=13.0.2`.
- **vLLM tree pin:** AEON **0.27.1** branch `aeon-v0.27.1` @ **`34aeb2303114bf3d8b2ecd582b7301868691735a`**. Upstream tag underneath is [v0.27.1](https://github.com/vllm-project/vllm/releases/tag/v0.27.1) (`6e448d0ea9` = v0.27.0 + #50424).
- **Kernels:** NVFP4 / FP8 for **12.0 / 12.0a**. `ENABLE_NVFP4_SM100=0`.
- **DSpark classes** are present in the 0.27.1 tree. **V2 model-runner, GB10, and UMA paths are not baked** as defaults.
- **Vision + audio extras** ship on this public slim tag (`vllm-audio-extra`, av, soundfile, soxr, librosa, torchaudio).
- Slim diet: other-arch FlashInfer cubins, CUDA statics, and Nsight removed. **B200/B300 (`sm_100` / `sm_103`) cubins were stripped.** If you need those, use a later **`:sm100` / `:sm103`** or the **`:sm120-fat`** image — not this slim tag.

Ada `8.9` is not a default. `native` is never the arch list.

---

## Hardware matrix (this slim tag only)

| Device | This `:sm120` slim |
|---|---|
| GeForce RTX 5090 (GB202, sm_120) | **Yes** |
| RTX PRO 6000 Blackwell (sm_120) | **Yes** |
| DGX Spark / GB10 / sm_121a / UMA | **No** — Spark Ultimate |
| B200 (sm_100) / B300 (sm_103) | **No** — cubins stripped; later `:sm100` / `:sm103` or fat |
| Ada 8.9 (4090 class) | **No** |

---

## Serve flags — Qwen3.8 judge (no YaRN, no MTP)

Hunt-judge seat: Qwen3.8, reasoning parser `qwen3`, tool-call parser `qwen3_coder`, auto tool choice. **No YaRN. No MTP.** Tight batching: `--max-num-seqs 4`, `--max-num-batched-tokens 16384`.

```bash
docker run --gpus all --ipc=host --shm-size=16g \
  -p 18081:18081 \
  --entrypoint vllm \
  ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120 \
  serve /model \
    --host 0.0.0.0 --port 18081 \
    --served-model-name judge \
    --dtype auto \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384
```

Do not add YaRN rope scaling. Do not add MTP / speculative config for this judge seat.

---

## Sibling vs Spark Ultimate

Same AEON 0.27.1 merge organs (NVFP4, FlashInfer trio, torch 2.13+cu130 pins) — **different silicon, different package**.

- RTX slim: dedicated GDDR, FLASHINFER-native, no UMA clamp, no GB10 `12.1a`.
- Spark: `ghcr.io/aeon-7/aeon-vllm-ultimate` aarch64 / sm_121a / UMA. Leave it alone.

---

## Host notes

Use **WSL Ubuntu-24.04 + docker.io**. Do not start Docker Desktop on Aorus (it fights pgembed `:5432`).

---

## License / credits

Apache-2.0 for this repo and the OCI license label. [vLLM](https://github.com/vllm-project/vllm) Apache-2.0. PyTorch BSD-3-Clause. FlashInfer / CUTLASS as upstream. AEON patches MIT.

This container is provided **AS IS**. You are solely responsible for prompts, outputs, and what you do with them.

AEON FORGE / @SpaceTimeViking.
