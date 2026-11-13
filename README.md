# AEON vLLM Ultimate RTX

[![docker](https://img.shields.io/badge/ghcr.io-aeon--7%2Faeon--vllm--ultimate--rtx-blue?logo=docker)](https://github.com/users/AEON-7/packages/container/package/aeon-vllm-ultimate-rtx)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1%2Baeon.sm120.rtx-orange)](https://github.com/vllm-project/vllm/releases/tag/v0.27.1)
[![arch](https://img.shields.io/badge/linux%2Famd64-CUDA%2013%20sm__120-green)](https://github.com/AEON-7/vllm-ultimate-rtx)

[![Buy me a coffee on LotusMake](https://lotusmake.com/badges/coffee-light.v2.svg)](https://lotusmake.com/support) [![Subscribe on lotusmake.com](https://lotusmake.com/badges/subscribe-light.v2.svg)](https://lotusmake.com/support?mode=tiers)

**Public register:** AEON FORGE / [@SpaceTimeViking](https://github.com/SpaceTimeViking)

Slim, linux/amd64 vLLM image for **NVIDIA Blackwell consumer / workstation `sm_120`**: GeForce **RTX 5090** and **RTX PRO 6000 Blackwell**. Compiled for CUDA 13 (`12.0` + `12.0a`) so NVFP4 / FP8 kernels actually land. Vision + audio extras are in this public-tag slim.

GHCR package **`ghcr.io/aeon-7/aeon-vllm-ultimate-rtx`** is **public**: anyone can pull it, no login needed.

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
# WSL Ubuntu-24.04 + docker.io (not Docker Desktop); no GHCR login needed (public package)
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

Hunt-judge seat: Qwen3.8, reasoning parser `qwen3`, tool-call parser `qwen3_coder`, auto tool choice. **No YaRN. No MTP.** Tight batching: `--max-num-seqs 4`, `--max-num-batched-tokens 2048` — **2048, not 16384**; see [WSL2 long-prompt engine fault](#wsl2-long-prompt-engine-fault) below.

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
    --max-num-batched-tokens 2048
```

Do not add YaRN rope scaling. Do not add MTP / speculative config for this judge seat.

---

## WSL2 long-prompt engine fault

**Verified 2026-08-26 — Aorus RTX 5090, WSL2 Ubuntu-24.04, driver 610.47, headless GPU,
Qwen3.8-27B NVFP4 (compressed-tensors).**

With a large `--max-num-batched-tokens`, the **whole engine dies** partway through the first long
prompt:

```
RuntimeError: CUDA driver error: device not ready
vllm.v1.engine.exceptions.EngineDeadError
```

**The fix:** `--max-num-batched-tokens 2048`.

### Why the traceback lies

The fault site moves between runs — we saw `flashinfer_mm_fp4`, then, after changing the linear
backend, an inductor MLP buffer (`empty_strided_cuda((s18, 17408))`). That is an **asynchronous
CUDA fault surfacing at the next synchronising call**, so the frame in the traceback is merely
whoever allocated next. Do not chase it.

### It is latent — short prompts hide it

A smoke test and a full 24-prompt category benchmark all passed before the first real ~24k-token
conversation killed the engine. **Test with a long prompt or you will ship this.**

### What we measured

| `--linear-backend` | `--max-num-batched-tokens` | largest prompt | result |
|---|---|---|---|
| `auto` (default) | 8192 | 24,452 | EngineDeadError at `mm_fp4` |
| `cutlass` | 8192 | 37,984 | EngineDeadError at MLP buffer |
| `cutlass` | **2048** | **106,927** | OK |
| `auto` (default) | **2048** | 34,927 | OK |

Chunk size is the only variable that mattered. **`--linear-backend cutlass` is not the fix** and is
not required — that was a wrong hypothesis, disproven by the last row.

Smaller chunks also **buy KV**: the profiler reported 137,240 tokens at 8192 vs **142,637** at 2048,
because the activation workspace shrinks. Prompts above `--max-model-len` are still rejected
cleanly with HTTP 400 and the engine stays up — that is correct behaviour, not this bug.

### Suspected mechanism

Windows WDDM keeps its TDR watchdog over WSL2 GPU work (`TdrLevel=3`, `TdrDelay=2s` — both unset,
i.e. default, on this host). One 8192-token prefill chunk through a 27B model appears to exceed
that budget, so the device is reset underneath the process. Raising `TdrDelay` is the host-side
alternative, but it is a machine-wide registry change needing a reboot; capping the chunk is
per-service and reversible.

**Native Linux hosts are not expected to need this** — the watchdog is a Windows/WDDM construct.
Unverified: we only have the WSL2 box.

---

## Sibling vs Spark Ultimate

Same AEON 0.27.1 merge organs (NVFP4, FlashInfer trio, torch 2.13+cu130 pins) — **different silicon, different package**.

- RTX slim: dedicated GDDR, FLASHINFER-native, no UMA clamp, no GB10 `12.1a`.
- Spark: `ghcr.io/aeon-7/aeon-vllm-ultimate` aarch64 / sm_121a / UMA. Leave it alone.

---

## Host notes

Use **WSL Ubuntu-24.04 + docker.io**. Do not start Docker Desktop on Aorus (it fights pgembed `:5432`).

---

## Support the work

AEON-7 models, drafters and tools are built and trained independently, on my own hardware. If they're useful to you, please consider supporting development.

Patreon wrongfully shut down our account without notice and never responded to our appeals, so we built something even better. You can now support the work on **LotusMake**: buy us a Lotus Tea, fund GPU hours, back a project you want to see done, or subscribe for early and full access to publications while supporting what we build. You can become a creator there too: join, apply to become a creator, and get paid for the work you do in public.

[![Buy me a coffee on LotusMake](https://lotusmake.com/badges/coffee-light.v2.svg)](https://lotusmake.com/support) [![Subscribe on lotusmake.com](https://lotusmake.com/badges/subscribe-light.v2.svg)](https://lotusmake.com/support?mode=tiers)

**[AEON FORGE on LotusMake → lotusmake.com/c/aeon-forge](https://lotusmake.com/c/aeon-forge)** · **[Subscribe for early and full access](https://lotusmake.com/support?mode=tiers&from=%2Fc%2Faeon-forge)**

Milestones unlock bigger work: reaching **500 paid supporters** will fund fine-tuning larger models and bigger project releases. Subscribers also get early access to new releases, such as the [AEON DFlash2 drafter](https://huggingface.co/AEON-7/AEON-DFlash2-Qwen3.8-27B).

---

## License / credits

Apache-2.0 for this repo and the OCI license label. [vLLM](https://github.com/vllm-project/vllm) Apache-2.0. PyTorch BSD-3-Clause. FlashInfer / CUTLASS as upstream. AEON patches MIT.

This container is provided **AS IS**. You are solely responsible for prompts, outputs, and what you do with them.

AEON FORGE / @SpaceTimeViking.
