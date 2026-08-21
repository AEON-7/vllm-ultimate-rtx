# AEON vLLM Ultimate RTX — RTX 5090 · Blackwell sm_120 · NVFP4
#
# THIS IMAGE IS NOT FOR DGX SPARK / GB10.
# Spark / sm_121a lives at github.com/AEON-7/vllm-ultimate-dgx-spark
# and ghcr.io/aeon-7/aeon-vllm-ultimate. Do not retag that package.
#
# Target: NVIDIA GeForce RTX 5090 (GB202) = consumer Blackwell sm_120.
# Compile arches (verified, not invented):
#   TORCH_CUDA_ARCH_LIST=12.0          # 5090 identity (sm_120)
#   also 12.0a                         # CUDA 13.2 nvcc accepts sm_120a /
#                                      # compute_120a. CUTLASS NVFP4 kernels
#                                      # historically #ifdef on the 'a' family
#                                      # (Spark needed 12.1a, not bare 12.1).
#   CMAKE_CUDA_ARCHITECTURES=120;120a
# NEVER: native, unset, 8.9 (Ada), 12.1 / 12.1a (GB10 / Spark), sm_100 (B200).
# NEVER bake DSpark / VLLM_USE_V2_MODEL_RUNNER / GB10 TRITON_ATTN defaults.
#
# Platform: linux/amd64. Spark is aarch64 — do not bake this image there.
#
# Source tree: COPY vllm-src/  (AEON v0.27.1 3-way merge). Pin in SOURCE.md.
# Image: ghcr.io/aeon-7/aeon-vllm-ultimate-rtx   tags :sm120 :5090 :rtx5090

FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- arch pins: RTX 5090 · Blackwell sm_120 · NVFP4 ---
ENV TORCH_CUDA_ARCH_LIST="12.0;12.0a"
ENV CMAKE_CUDA_ARCHITECTURES="120;120a"
ENV CUDAARCHS="120;120a"
ENV FLASHINFER_CUDA_ARCH_LIST="120;120a"
ENV CUTLASS_NVCC_ARCHS="120;120a"
ENV ENABLE_NVFP4_SM100=0
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0
ENV CCACHE_DISABLE=1
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=12
ENV NVCC_THREADS=2
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.27.1+aeon.sm120.rtx"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.27.1+aeon.sm120.rtx"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# NOTE: VLLM_USE_V2_MODEL_RUNNER deliberately NOT set.
# NOTE: no DSpark / GB10 / UMA env baked.

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# jammy has no python3.12 in archive — deadsnakes. Do not use jammy python3-pip
# (that is 3.10 and would shadow /usr/bin/pip).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates curl ffmpeg gnupg software-properties-common \
        gcc-12 g++-12 make ninja-build \
        libsndfile1 libgomp1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/gcc-12 /usr/bin/gcc && \
    ln -sf /usr/bin/g++-12 /usr/bin/g++ && \
    ln -sf /usr/bin/gcc-12 /usr/bin/cc && \
    ln -sf /usr/bin/g++-12 /usr/bin/c++ && \
    python3.12 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade pip && \
    ln -sf "$(command -v pip3 || true)" /usr/local/bin/pip 2>/dev/null || true && \
    python3 -m pip --version && python3 --version

ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=g++-12
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# Fail the bake if anyone smuggled Spark 12.1a or left the list native/unset.
RUN python3 -c "import os,sys; tcl=os.environ.get('TORCH_CUDA_ARCH_LIST',''); cmake=os.environ.get('CMAKE_CUDA_ARCHITECTURES',''); print('TORCH_CUDA_ARCH_LIST=',tcl); print('CMAKE_CUDA_ARCHITECTURES=',cmake); \
bad = (not tcl) or tcl.strip().lower()=='native' or '12.1' in tcl or '121' in cmake or '8.9' in tcl or ('12.0' not in tcl) or ('120' not in cmake); \
sys.exit('bad arch pins for 5090') if bad else print('arch pins OK: RTX 5090 · Blackwell sm_120 · NVFP4')"

# CUDA 13.0 toolkit assertion + sm_120 / sm_120a accepted by this nvcc.
RUN nvcc --version && \
    python3 -c "import subprocess,sys; out=subprocess.check_output(['nvcc','--help'], text=True, stderr=subprocess.STDOUT); need=['sm_120','sm_120a','compute_120','compute_120a']; missing=[n for n in need if n not in out]; \
sys.exit('nvcc missing: '+', '.join(missing)) if missing else print('nvcc accepts sm_120 and sm_120a (verified, not invented)')"

# torch 2.13.0+cu130 — same pin as Spark 0.27.1, x86_64 wheels.
RUN pip install --no-cache-dir "torch==2.13.0" "torchvision==0.28.0" \
        --index-url https://download.pytorch.org/whl/cu130 && \
    python3 -c "import torch,triton; assert torch.__version__.startswith('2.13.0'); assert torch.version.cuda.startswith('13.0'); assert triton.__version__.startswith('3.7'); print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.version.cuda)"

RUN pip install --no-cache-dir "nvidia-nccl-cu13==2.30.7" && \
    python3 -c "import importlib.metadata as m; v=m.version('nvidia-nccl-cu13'); assert v=='2.30.7', v; print('nccl', v)"

COPY vllm-src/ /build/vllm-src/

WORKDIR /build/vllm-src
RUN pip install --no-cache-dir "cmake>=3.26.1" "setuptools-rust>=1.9.0" "setuptools-scm>=8" wheel && \
    python3 use_existing_torch.py

# Compile CUDA/CUTLASS/NVFP4 for sm_120 + sm_120a only.
RUN CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
        TORCH_CUDA_ARCH_LIST="12.0;12.0a" \
        CMAKE_CUDA_ARCHITECTURES="120;120a" \
        CUDAARCHS="120;120a" \
        FLASHINFER_CUDA_ARCH_LIST="120;120a" \
        CUTLASS_NVCC_ARCHS="120;120a" \
        ENABLE_NVFP4_SM100=0 \
        pip install --no-deps --no-build-isolation -v . \
        2>&1 | tee /tmp/vllm-install.log | tail -120

# CUTLASS / quack / tvm-ffi / tilelang — same ABI trio as Spark 0.27.1.
RUN pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.6.0" "quack-kernels==0.6.1" \
        "apache-tvm-ffi==0.1.11" "tilelang==0.1.12" && \
    python3 -c "import importlib.metadata as m; c=m.version('nvidia-cutlass-dsl'); q=m.version('quack-kernels'); assert c.startswith('4.6'), c; assert q=='0.6.1', q; print('cutlass-dsl', c, '| quack', q)"

# FlashInfer 0.6.16.post3 trio — x86_64 cu130 (NOT the aarch64 Spark jit-cache).
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin || true; \
    pip install --no-cache-dir \
        --extra-index-url https://flashinfer.ai/whl/ \
        --extra-index-url https://flashinfer.ai/whl/cu130/ \
        "flashinfer-python==0.6.16.post3" "flashinfer-cubin==0.6.16.post3" \
        "flashinfer-jit-cache==0.6.16.post3" && \
    python3 -c "import flashinfer, importlib.metadata as m; v=flashinfer.__version__; assert v.startswith('0.6.16.post3'), v; print('flashinfer trio', v, m.version('flashinfer-cubin'), m.version('flashinfer-jit-cache'))"

RUN pip install --no-cache-dir "torchcodec==0.16.0" && \
    python3 -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec', torchcodec.__version__)" \
    || (pip uninstall -y torchcodec && echo "[WARN] torchcodec removed")

RUN pip install --no-cache-dir "xgrammar>=0.2.1,<1.0.0" "scipy>=1.11" \
        "transformers==5.14.1" && \
    python3 -c "import transformers; from xgrammar import normalize_tool_choice; print('transformers', transformers.__version__)"

RUN pip install --no-cache-dir --no-deps \
        "turboquant @ git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers" \
        || echo "[WARN] turboquant install attempted"

COPY humming-stub/ /tmp/humming-stub/
RUN pip install --no-cache-dir /tmp/humming-stub && rm -rf /tmp/humming-stub

COPY verify.py /tmp/verify.py
RUN python3 /tmp/verify.py && rm /tmp/verify.py

WORKDIR /
RUN python3 -c "import vllm,inspect,os; assert vllm.__version__.startswith('0.27.1'); import vllm.v1.attention.backends.triton_attn as t; assert 'nvfp4' in inspect.getsource(t).lower(); import vllm.utils.torch_utils as tu; assert 'as_strided' not in tu._nvfp4_split_data_scale.__code__.co_names; assert os.environ.get('VLLM_USE_V2_MODEL_RUNNER') is None; assert os.environ.get('TORCH_CUDA_ARCH_LIST')=='12.0;12.0a'; print('vllm', vllm.__version__, 'sm_120 RTX gate OK')"

COPY nvfp4_kv_gate.py /tmp/nvfp4_kv_gate.py
RUN python3 /tmp/nvfp4_kv_gate.py && rm /tmp/nvfp4_kv_gate.py

RUN rm -rf /build /root/.cache/pip

LABEL org.opencontainers.image.title="AEON vLLM Ultimate RTX (5090 / sm_120)" \
      org.opencontainers.image.description="AEON vLLM Ultimate RTX — vLLM 0.27.1 built from source for RTX 5090 / Blackwell sm_120 / NVFP4. Dedicated-VRAM consumer Blackwell. NOT for DGX Spark / GB10 / sm_121a." \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-rtx" \
      org.opencontainers.image.documentation="https://github.com/AEON-7/vllm-ultimate-rtx" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="AEON-7" \
      org.opencontainers.image.version="0.27.1+aeon.sm120.rtx" \
      ai.aeon.vllm_base="vLLM 0.27.1 (from-source, sm_120 + sm_120a)" \
      ai.aeon.hardware="NVIDIA GeForce RTX 5090 GB202 SM120" \
      ai.aeon.arch="sm_120" \
      ai.aeon.torch_cuda_arch_list="12.0;12.0a" \
      ai.aeon.features="rtx5090,sm120,nvfp4,nvfp4-kv,nvfp4-awq,flashinfer-0.6.16.post3,torch-2.13.0,cutlass-dsl-4.6.0,nccl-2.30.7,dflash-optional,no-dspark-default,no-gb10-flags" \
      ai.aeon.not_for="DGX Spark GB10 sm_121a"

ENTRYPOINT ["/bin/bash"]
