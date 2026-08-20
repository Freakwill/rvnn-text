"""Save and load trained RvNN text models."""

from __future__ import annotations

from pathlib import Path

import torch

from .grammar import Grammar
from .model import RvNNText


def save_checkpoint(model: RvNNText, path: str | Path) -> Path:
    """Save the model weights, grammar and hyper-parameters to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dim": model.dim,
            "state_dict": model.state_dict(),
            "grammar": model.grammar.to_dict(),
        },
        path,
    )
    return path


def load_model(path: str | Path, device: torch.device | None = None) -> RvNNText:
    """Load a model from a checkpoint produced by :func:`save_checkpoint`."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    grammar = Grammar.from_dict(payload["grammar"])
    model = RvNNText(grammar, dim=payload["dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device)
