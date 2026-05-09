from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MANO_C(nn.Module):
    """
    Minimal stub for legacy MANO model objects referenced by pickles.

    It stores state but does not implement forward logic. This is only
    intended to satisfy pickle loading for visualization scripts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._raw_args = args
        self._raw_kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_raw_state"] = state

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("MANO_C stub does not implement forward")
