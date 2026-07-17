"""State-dict checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    """Atomically save a state-dict checkpoint, never a pickled model object."""
    if "model_state_dict" not in state:
        raise KeyError("checkpoint state must contain model_state_dict")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load and minimally validate a V9 checkpoint."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    state = torch.load(source, map_location=device, weights_only=False)
    required = {"model_state_dict", "epoch", "config"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Checkpoint {source} is missing fields: {missing}")
    return state

