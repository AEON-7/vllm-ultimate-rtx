"""humming.schema.fp8 stub (permissive)."""

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

