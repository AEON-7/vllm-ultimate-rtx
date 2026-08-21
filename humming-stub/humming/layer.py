"""humming.layer stub (permissive)."""


class HummingLayerMeta(type):
    """Stub metaclass."""
    pass


class HummingMethod:
    """Stub for the humming layer method. Constructed only when humming quant is
    actually SELECTED — which the AEON fleet (modelopt / compressed-tensors)
    never does — so raising here is the correct loud failure."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "humming-stub: real humming library is not installed. "
            "Choose a different quantization method (e.g. modelopt, awq, gguf)."
        )


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

