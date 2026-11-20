"""
Burstormer teacher adapter for output-level KD.

Location: models/burstormer/burstormer_teacher.py

Role:
    - Wrap the original Burstormer (from models/burstormer/Network.py) as a frozen teacher.
    - Enforce eval mode + no gradient + no_grad() in forward.
    - Reject batch_size != 1 explicitly, because Burstormer.forward does `burst = burst[0]`.

NOT responsible for:
    - Building sigma_estimate (real .npz data is handled by BurstDataset;
      the synthetic smoke dataset uses a placeholder).
    - Loading datasets or test sets.
    - Computing PSNR/SSIM/LPIPS (TODO: belongs to evaluation).
    - Post-processing (DenoisingPostProcess). TODO.

Requires:
    torch, torchvision, pytorch-lightning, einops installed.
"""

from pathlib import Path

import torch
import torch.nn as nn

# Relative import: '.' = models/burstormer/. Network.py must sit next to this file.
from .Network import Burstormer


# Default checkpoint name, matching the file you downloaded from the official link.
DEFAULT_CHECKPOINT_NAME = "Burstormer.ckpt"

# Project root = kd_project/ (three levels up from this file):
#   .../kd_project/models/burstormer/burstormer_teacher.py
#         ^1       ^2         ^3
# .parent          -> models/burstormer/
# .parent.parent   -> models/
# .parent.parent.parent -> 3rd_KD_Demo/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BurstormerTeacher(nn.Module):
    """Frozen Burstormer wrapper.

    Input contract:
        burst_6ch: torch.Tensor, shape [1, K, 6, H, W]
                   K = 8 (hard-coded in Burstormer.rec_conv1 via num_features*8)
                   6 = 3 RGB channels + 3 noise-estimate channels
                   (noise estimate is built by the data loader)

    Output contract:
        out: torch.Tensor, shape [1, 3, H, W]
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", mode: str = "color"):
        super().__init__()

        if mode != "color":
            raise NotImplementedError(
                "This project only wires up color denoising. "
                "Grayscale would require in_chn=2 / out_chn=1 and different data."
            )

        # load_from_checkpoint is a PyTorch Lightning classmethod.
        model = Burstormer.load_from_checkpoint(
            checkpoint_path, 
            mode=mode,
            weights_only=False  # 明確允許載入非權重物件
        )

        model.eval()
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.device = device

    def train(self, mode: bool = True):
        """Override train() so the teacher NEVER leaves eval mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, burst_6ch: torch.Tensor) -> torch.Tensor:
        if burst_6ch.shape[0] != 1:
            raise ValueError(
                f"Burstormer requires batch_size=1 "
                f"(its forward does `burst = burst[0]`). "
                f"Got batch_size={burst_6ch.shape[0]}."
            )
        if burst_6ch.shape[2] != 6:
            raise ValueError(
                f"Burstormer expects 6 input channels (3 RGB + 3 noise estimate). "
                f"Got {burst_6ch.shape[2]} channels."
            )
        with torch.no_grad():
            out = self.model(burst_6ch)
        return out


def create_teacher(device, checkpoint_path=None):
    """Factory function called by run_kd_smoke.py via --teacher-factory.

    Import path: models.burstormer.burstormer_teacher:create_teacher

    The checkpoint path is resolved relative to the project root
    (kd_project/), NOT the current working directory. This means you
    can run `python run_kd_smoke.py` from any directory and the path
    will still resolve correctly.
    """
    if checkpoint_path is None:
        checkpoint_path = _PROJECT_ROOT / "weights" / DEFAULT_CHECKPOINT_NAME
    return BurstormerTeacher(checkpoint_path=str(checkpoint_path), device=device)