from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionHOT3D(Dataset):
    """
    Minimal stub used to unpickle legacy MotionHOT3D datasets.

    The original class lives in an external repo. We keep a permissive
    container so pickle can restore stored attributes and examples.
    """

    def __init__(self, examples: list[Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.examples = examples or []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            # Preserve opaque state for debugging; keep object usable.
            self.__dict__["_raw_state"] = state

    def __len__(self) -> int:
        container = _resolve_container(self)
        return len(container) if container is not None else 0

    def __getitem__(self, idx: int) -> Any:
        container = _resolve_container(self)
        if container is None:
            raise IndexError("No data container found in MotionHOT3D stub")
        return container[idx]


def _resolve_container(dataset: MotionHOT3D) -> list[Any] | tuple[Any, ...] | np.ndarray | torch.Tensor | None:
    # Common field names seen in pickled datasets.
    for name in ("examples", "items", "data", "samples", "entries", "clips", "records", "frames"):
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                return value
    return None
