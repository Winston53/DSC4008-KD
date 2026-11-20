"""NAFNet student adapter for the KD project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import nafnet_arch

_MODEL_CONFIG: dict[str, Any] = {
    "img_channel": 3,
    "width": 32,
    "enc_blk_nums": [2, 2, 4, 8],
    "middle_blk_num": 12,
    "dec_blk_nums": [2, 2, 2, 2],
}


class NAFNetStudent(nn.Module):
    """Adapt the project's NAFNet architecture to the student model contract.

    Input and output are RGB tensors with shape ``[B, 3, H, W]``. The wrapper
    does not clamp or normalize values; dataset, teacher, and target ranges
    must be aligned by the data/training pipeline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nafnet_arch.NAFNet(**_MODEL_CONFIG)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "NAFNetStudent expects an RGB tensor shaped [B, 3, H, W]; "
                f"received {tuple(image.shape)}."
            )
        return self.model(image)


def _load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NAFNet checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Expected a state_dict or a checkpoint containing "
            "'model_state_dict'."
        )
    model.load_state_dict(state_dict)


def create_student(
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
) -> nn.Module:
    """Create the trainable NAFNet student.

    The architecture matches the NAFNet configuration in the source
    ``train.py``. By default weights are randomly initialized, as in that
    training script. Optionally load its ``nafnet_best.pt`` checkpoint.

    Factory path: ``models.nafnet.nafnet_student:create_student``
    """
    model = NAFNetStudent()
    if checkpoint_path is not None:
        _load_checkpoint(model.model, checkpoint_path)
    return model.to(device)
