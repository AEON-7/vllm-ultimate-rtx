"""Self-contained verification of the aeon-vllm-ultimate-rtx image."""
import os
import sys

print("=" * 60)
print("AEON vLLM Ultimate RTX — RTX 5090 · Blackwell sm_120 · NVFP4")
print("=" * 60)

tcl = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
print(f" TORCH_CUDA_ARCH_LIST: {tcl}")
if "12.0" not in tcl:
    print(" TORCH_CUDA_ARCH_LIST must contain 12.0 (sm_120)")
    sys.exit(1)
if "12.1" in tcl:
    print(" 12.1 leaked — that is Spark/GB10, not 5090")
    sys.exit(1)
if tcl.strip().lower() == "native" or not tcl:
    print(" arch list unset/native — refuse")
    sys.exit(1)

if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None:
    print(" VLLM_USE_V2_MODEL_RUNNER is set — not allowed on the 5090 default path")
    sys.exit(1)

try:
    import vllm
    print(f" vllm: {vllm.__version__}")
except Exception as e:
    print(f" vllm: FAIL ({e})")
    sys.exit(1)

try:
    import flashinfer
    print(f" flashinfer: {flashinfer.__version__}")
except Exception as e:
    print(f" flashinfer: WARN ({e})")

try:
    import torch
    print(f" torch: {torch.__version__} (cuda {torch.version.cuda})")
    print(f" cuda avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f" device cap: {cap[0]}.{cap[1]}  (5090 is 12.0 / sm_120)")
        name = torch.cuda.get_device_name(0)
        print(f" device: {name}")
        if cap != (12, 0):
            print(" WARN: this box is not sm_120 — image is still the 5090 lane")
except Exception as e:
    print(f" torch: FAIL ({e})")
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig
    print(" vllm.LLM: importable")
except Exception as e:
    print(f" vllm.LLM: FAIL ({e})")
    sys.exit(1)

print()
print("GREEN - aeon-vllm-ultimate-rtx ready (sm_120)")
