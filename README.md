# AEON vLLM Ultimate RTX — RTX 5090 · Blackwell sm_120 · NVFP4

[![docker](https://img.shields.io/badge/ghcr.io-aeon--7%2Faeon--vllm--ultimate--rtx-blue?logo=docker)](https://ghcr.io/aeon-7/aeon-vllm-ultimate-rtx)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1%2Baeon.sm__120.rtx-orange)](https://github.com/vllm-project/vllm)
[![sm_120](https://img.shields.io/badge/sm__120-RTX%205090-green)](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

**Public register:** AEON FORGE / [@SpaceTimeViking](https://github.com/SpaceTimeViking)

The Spark image is a 121 GB unified-memory poem. The 5090 is a 32 GB dedicated-VRAM knife. Same Blackwell marketing name, **different silicon**. We stopped pretending one fat image could serve both.

| | This repo | The other one |
|---|---|---|
| **Repo** | [AEON-7/vllm-ultimate-rtx](https://github.com/AEON-7/vllm-ultimate-rtx) | [AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark) |
| **Image** | `ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120` | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` |
| **GPU** | GeForce RTX 5090 (GB202) | DGX Spark (GB10) |
| **Arch** | **sm_120** · `TORCH_CUDA_ARCH_LIST=12.0` (+ `12.0a`) | sm_121a · `12.1a` |
| **ISA / box** | linux/amd64, dedicated GDDR7 | linux/aarch64, 121 GB UMA |
| **Attention default** | **FLASHINFER** | TRITON_ATTN (GB10) |
| **DSpark / V2 pin** | not baked | Spark-native |

**This image is NOT for GB10 / DGX Spark.** If you are on Spark, leave. The sm_121a recipe is the sibling repo. Compiling only `12.1a` and hoping it runs on a 5090 is how you ship Marlin fallbacks and dead kernels. We do not do that here.

> ⚠️ **Ada `8.9` is not a default.** Do not add it "just in case." 5090 is sm_120. Period.

---

## Why the split (read this before you retag `:latest`)

vLLM's CUTLASS NVFP4 path is `#ifdef`'d on the architecture-specific family tag. On Spark we learned the hard way: **bare `12.1` strips the FP4 kernels and you fall back to Marlin.** The 5090 analogue is `12.0` vs `12.0a`.

- **RTX 5090 (GB202) = compute capability 12.0 = `sm_120`.** Confirmed by NVIDIA / PyTorch (`CUDA capability sm_120`) and by `nvidia-smi` on the Aorus box.
- **DGX Spark (GB10) = `sm_121a`.** Different chip. Different OS ABI (SBSA vs x86_64). Different memory model (UMA vs dedicated VRAM).
- CUDA 13.2 `nvcc --help` on the AEON toolchain **lists** `sm_120`, `sm_120a`, `compute_120`, `compute_120a`. `120a` is not invented; it is accepted. We compile it because that is the family tag CUTLASS NVFP4 historically gates on.
- `TORCH_CUDA_ARCH_LIST` is **never** unset and **never** `native`. `native` bakes whatever box you happened to stand on. Standing on Spark would bake sm_121a into a 5090 image. That is a bug.

```
TORCH_CUDA_ARCH_LIST=12.0;12.0a
CMAKE_CUDA_ARCHITECTURES=120;120a
CUDAARCHS=120;120a
FLASHINFER_CUDA_ARCH_LIST=120;120a
CUTLASS_NVCC_ARCHS=120;120a
ENABLE_NVFP4_SM100=0
```

GHCR package is **`aeon-vllm-ultimate-rtx`**. Tags `:sm120` `:5090` `:rtx5090`. We do not overwrite `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`.

---

## What we kept from the Spark stack

The 0.27.1 AEON tree — the same 3-way merge that made Spark fast and *correct* — still matters on a 5090. We keep the organs, we drop the GB10 costume.

| Keep | Why it still hits on consumer Blackwell |
|---|---|
| **vLLM 0.27.1 from source** | `aeon-v0.27.1` @ `34aeb2303114bf3d8b2ecd582b7301868691735a` |
| **NVFP4 weights** | ModelOpt / compressed-tensors FP4 on Blackwell tensor cores. `--linear-backend flashinfer_cutlass` when the checkpoint has per-input global scales. |
| **NVFP4_AWQ** | `quant_algo=NVFP4_AWQ` + `pre_quant_scale` (Gemma-4-31B-DECKARD class). |
| **Triton NVFP4-KV (NHD-safe)** | PR #44389 carry. Upstream `as_strided` still corrupts NHD. Build-gated by `nvfp4_kv_gate.py`. `--kv-cache-dtype nvfp4` |
| **FlashInfer 0.6.16.post3 trio** | python + cubin + jit-cache, **x86_64 cu130** wheels (not the Spark aarch64 jit-cache). |
| **torch 2.13.0+cu130 / Triton 3.7.1** | Same pin as Spark 0.27.1, different wheel ISA. |
| **cutlass-dsl 4.6.0 + quack 0.6.1 + tvm-ffi 0.1.11** | Do not bump cutlass to 4.7 — quack 0.6.1 hard-pins 4.6. |
| **NCCL 2.30.7** | Same floor. TP=1 is the 5090 default (one card). |
| **DFlash SWA + #41703 ctx-slot mask** | Optional. Long-run chat/student later. Not the judge default. |
| **TurboQuant CUDA-graph-safe fork** | Opt-in `--kv-cache-dtype tq_k8v4`. Don't mix with NVFP4-KV. |
| **thread_local CUDA-graph capture (#48053)** | Still the right capture mode. |
| **transformers 5.14.1, torchcodec 0.16.0, xgrammar** | Fleet classes + video + tool-choice. |

## What we dropped (on purpose)

| Drop | Why it is poison on a 5090 |
|---|---|
| **`TORCH_CUDA_ARCH_LIST=12.1a`** | That is GB10. Compiling only sm_121a and hoping is how you miss the 5090 cubin. |
| **DSpark as a default** | Markov-head speculation is a Spark/MRv2 research path. Not the 5090 judge seat. |
| **`VLLM_USE_V2_MODEL_RUNNER=0/1` baked** | Do not stuff the V2 pin onto this path. Let vLLM route. Re-pin per-service only if you know why. |
| **`TRITON_ATTN` as the default** | That was the GB10 winner (FlashInfer NVFP4-KV wanted SM100; Triton was the software path). On a 5090, **FLASHINFER** is the native attention / NVFP4 partner. Triton stays in the image as a fallback. |
| **UMA clamps / `gpu-memory-utilization 0.60`** | Spark shares LPDDR5X with the CPU. The 5090 does not. Dedicated 32 GB — run the card. |
| **`VLLM_ENABLE_STARTUP_PLAN` panic** | That was an NVML-on-UMA landmine. 5090 NVML is a real device. |
| **aarch64 / sbsa CUDA paths** | This image is linux/amd64. |
| **MTP=2 as the image CMD** | Judge seat prefills long and decodes tiny JSON. MTP is a later long-run chat/student knob. |

---

## Quickstart (RTX 5090, WSL Ubuntu-24.04)

Host notes, because this is how the Aorus actually lives:

```bash
# WSL Ubuntu-24.04 — GPU must show up here
nvidia-smi
# NVIDIA GeForce RTX 5090     (compute capability 12.0 / sm_120)

# Do NOT start Docker Desktop on Aorus. It collides with pgembed :5432.
# Use a docker engine that is ALREADY running (native WSL docker, or pull
# a pre-baked GHCR image). Starting Docker Desktop is out of scope.
```

Pull (once an image has been baked — see [Bake](#bake-linuxamd64--sm_120)):

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120
# aliases: :5090  :rtx5090
```

### Recipe J — tight-KV judge (default seat, no MTP)

Port **18081** (MODELFORGE). Not 8000 if Hermes / a Brev tunnel already owns it. Short context, thinking off, high concurrency, NVFP4 weights + NVFP4-KV, FLASHINFER.

```bash
docker run -d --name aeon-vllm-rtx-judge \
    --gpus all --ipc=host --shm-size=16g \
    -p 18081:18081 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -v /models/judge:/model:ro \
    --entrypoint vllm \
    ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120 \
    serve /model \
        --host 0.0.0.0 --port 18081 \
        --served-model-name judge \
        --dtype auto \
        --quantization modelopt \
        --kv-cache-dtype nvfp4 \
        --attention-backend FLASHINFER \
        --linear-backend flashinfer_cutlass \
        --max-model-len 8192 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --generation-config vllm \
        --trust-remote-code
```

Compose: [`docker-compose.judge.yml`](docker-compose.judge.yml).

Thinking stays **off** at the request (judge JSON, not a chain-of-thought essay):

```bash
curl -s http://127.0.0.1:18081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "judge",
    "messages": [{"role":"user","content":"Score this. Return JSON only."}],
    "max_tokens": 128,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**Why these flags:** `--kv-cache-dtype nvfp4` is the capacity play on 32 GB; `--attention-backend FLASHINFER` is the 5090-native winner; `--max-model-len 8192` + `--max-num-seqs 64` is a tight-KV judge (many short JSON decodes, not one 200k chat); `--gpu-memory-utilization 0.90` is legal because this is dedicated VRAM, not Spark UMA; **no** `--speculative-config`, **no** MTP.

### Recipe L — long-run chat / student (later, not this bake's CMD)

When you actually want a conversation, not a judge:

- Raise `--max-model-len` (32k–128k as VRAM allows under NVFP4-KV).
- Drop `--max-num-seqs` to 4–16.
- Optional DFlash: `--speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":12,"attention_backend":"FLASHINFER"}'`
- **MTP=2** is a knob for that seat (`qwen3_5_mtp`, `num_speculative_tokens: 2`). It is **not** the image default and it is **not** the judge seat. Prefills are long; judge decodes are tiny. Do not waste draft tokens on a 40-token JSON object.

---

## What's inside

| Component | Pin | Why |
|---|---|---|
| **vLLM** | 0.27.1+aeon.sm120.rtx | AEON 3-way merge, compiled for **sm_120 + sm_120a** |
| **PyTorch** | 2.13.0+cu130 | CUDA 13.0, x86_64 |
| **FlashInfer** | 0.6.16.post3 trio | NVFP4 GEMM + attention, x86_64 cu130 |
| **CUTLASS-DSL** | 4.6.0 | quack 0.6.1 hard-pin |
| **NCCL** | 2.30.7 | same floor as Spark 0.27.1 |
| **transformers** | 5.14.1 | fleet classes |
| **Arch list** | `12.0;12.0a` | 5090 identity + verified NVFP4 'a' family |

`ENABLE_NVFP4_SM100=0` — we do not compile B200-only kernels.

---

## WSL Ubuntu-24.04 host

The Aorus box runs **WSL Ubuntu-24.04**. That is the host this image is for.

```bash
wsl -d Ubuntu-24.04 -- nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 5090
```

Rules of the house:

1. **Do not start Docker Desktop.** It fights pgembed on `:5432` and the WSL docker CLI you see (`/mnt/c/Program Files/Docker/...`) is the Desktop stub. If `docker info` tells you to "activate WSL integration in Docker Desktop," that is a no.
2. Bake where Docker **already exists**: GitHub Actions `linux/amd64`, a native WSL docker engine that is already up, or another x86_64 box. **Not** spark-ecdd (aarch64). **Not** by launching Desktop.
3. Do not start a Qwen3.8-27B hunt on this GPU as part of the bake. Do not bounce friend Brev GPUs. Do not Hub-push models from this recipe.

---

## Bake (linux/amd64 · sm_120)

Recipe is landed. A full from-source CUDA compile is multi-hour and needs an x86_64 docker that is already running.

```bash
# 1) AEON v0.27.1 tree (same merge as Spark). On spark-ecdd this is:
#    /home/albert/aeon-builds/vllm-ultimate-v0.27.1/vllm-src
#    branch aeon-v0.27.1  SHA 34aeb2303114bf3d8b2ecd582b7301868691735a
#    Copy or bind that tree to ./vllm-src  (gitignored).

# 2) On an x86_64 host with Docker already running:
IMAGE=ghcr.io/aeon-7/aeon-vllm-ultimate-rtx \
TAG=sm120 \
VLLM_SRC=/path/to/aeon-v0.27.1/vllm-src \
  ./scripts/bake-rtx5090.sh

# 3) Push to GHCR (does not touch the Spark package)
docker push ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:sm120
docker push ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:5090
docker push ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:rtx5090
```

The bake script **refuses to run on aarch64**. Spark will not accidentally compile this.

Manual Actions workflow: `.github/workflows/image.yml` (`workflow_dispatch`). Default GHCR runners are usually too small for the compile; treat it as the documented command, not a promise.

Slim rebase (`Dockerfile.slim`) is a second step, after `:sm120-fat` exists — same story as Spark's 50 GB → 18 GB diet, but **amd64**.

---

## Provenance

See [SOURCE.md](SOURCE.md). Short version:

- Sibling of [vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark) (Apache-2.0 labels, same AEON 0.27.1 tree).
- vLLM Apache-2.0. PyTorch BSD-3-Clause. TurboQuant Apache-2.0. AEON patches MIT.
- We do not vendor the 140 MB `vllm-src` tree. Pin it, compile it, throw the objects away.

---

## License

Apache-2.0 for this repo. vLLM Apache-2.0. PyTorch BSD-3-Clause. TurboQuant Apache-2.0. AEON patches MIT.

This container is provided **AS IS**. You are solely responsible for prompts, outputs, and what you do with them. If you do not accept that, do not run it.

---

## ☕ Support the work

AEON FORGE / @SpaceTimeViking. Tips go to more silicon, more open images, fewer "it works on my Spark" lies.

- BTC `bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4`
- ETH `0x1512667F6D61454ad531d2E45C0a5d1fd82D0500`
- SOL `DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t`
