"""Utilities for keeping optional PyTorch state local to model fitting."""

from __future__ import annotations

from functools import wraps

try:
    import torch
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None


def preserve_torch_rng(method):
    """Restore PyTorch's process-wide RNG state after seeded model fitting."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if torch is None or getattr(self, "random_state", None) is None:
            return method(self, *args, **kwargs)
        with torch.random.fork_rng(devices=None):
            return method(self, *args, **kwargs)

    return wrapped
