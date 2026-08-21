"""humming.config stub (permissive)."""


class GemmType:
    """Stub marker; never used for real compute by the AEON fleet."""
    pass


class WeightScaleType:
    """Stub marker (referenced by v0.25.0 humming facade)."""
    pass


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

