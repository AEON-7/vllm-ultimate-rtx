"""humming.dtypes stub.

vLLM's humming quant path imports dtype descriptors from here at
quant-registry-build time (e.g. `dtypes.float4e2m1`, `dtypes.uint4`,
`dtypes.d_dtypes`). We ship a STUB because the AEON fleet uses modelopt /
compressed-tensors NVFP4, never humming quant — but the registry-build import
must not crash. A module-level ``__getattr__`` resolves ANY attribute to a
stable, harmless sentinel so this survives future humming API growth without
whack-a-mole. Sentinels are identity-stable (safe as dict keys) and never carry
out real compute; actually selecting a humming quant method still fails loudly
via HummingMethod / DataType elsewhere.
"""

from typing import Any


class _DTypeSentinel:
    """A placeholder for a humming dtype descriptor. Hashable, identity-stable,
    supports the incidental attribute/append access registry code performs."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:  # noqa: D401
        return f"<humming-stub dtype {self._name!r}>"

    def append(self, *_a: Any, **_k: Any) -> None:
        return None

    def __iter__(self):
        return iter(())

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(self._name)


class DataType:
    """Placeholder. Import-time construction is tolerated (some registry paths
    instantiate it); actual humming compute is never reached by the AEON fleet."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Do NOT raise here — v0.25.0 builds the quant registry at engine init,
        # which may construct descriptors. Selecting humming quant for real is
        # gated elsewhere (HummingMethod), which our fleet never triggers.
        self._args = args
        self._kwargs = kwargs


# `d_dtypes` is referenced as a real list (registry appends to it).
d_dtypes: list = []

_CACHE: dict = {}


def __getattr__(name: str) -> Any:
    """Resolve any humming dtype name (float4e2m1, uint2/3/4, float8e8m0, …) to a
    stable sentinel so quant-registry imports complete without the real lib."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    obj = _CACHE.get(name)
    if obj is None:
        obj = _DTypeSentinel(name)
        _CACHE[name] = obj
    return obj
