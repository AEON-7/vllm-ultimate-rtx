"""humming.schema stub package (permissive).

v0.25.0 references many schema classes (AWQ, Bitnet, CompressedTensors*, Fp8*,
Mxfp4*, GptOss*, GPTQ*, Modelopt* in submodules). All resolve to harmless
placeholder classes via the catch-all so the quant registry imports cleanly.
"""


class BaseInputSchema: pass
class BaseWeightSchema: pass
class HummingInputSchema(BaseInputSchema): pass
class HummingWeightSchema(BaseWeightSchema): pass


_CACHE: dict = {}

def __getattr__(name: str):
    """Resolve any requested humming symbol to a cached, harmless placeholder
    class so vLLM's eager quant-registry imports succeed without the real lib.
    Selecting a humming quant method for real still fails via HummingMethod."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    obj = _CACHE.get(name)
    if obj is None:
        obj = type(name, (), {"__init__": lambda self, *a, **k: None})
        _CACHE[name] = obj
    return obj

