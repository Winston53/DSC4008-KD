"""
BurstDataset: consume the .npz files produced by the groupmate's dataset pipeline.

Role:
    - Read .npz (truth, noisy, sig_read, sqrt_sig_shot, white_level).
    - Compute sigma_estimate and build the 6-channel teacher input.
    - Return {teacher_input, reference_rgb, synthetic_target}.

NOT responsible for:
    - Synthesizing bursts (that is the groupmate's job; see their
      burst_synthesis.py).
    - Splitting train/val/test (the groupmate provides index ranges).
    - Post-processing (DenoisingPostProcess) at evaluation time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------
# Block 1: sigma estimate formula (single source of truth)
# --------------------------------------------------------------------
# WHY a standalone function:
#   The .npz loader uses this formula for real-data samples. The synthetic
#   smoke dataset intentionally uses a constant placeholder instead.
#   Keeping the real-data calculation here makes it independently testable.
#
# WHY clamp(min=0.0):
#   Shot noise is proportional to pixel value; negative values have no
#   physical meaning for shot noise. Matches the official
#   color_denoise_test_set.py implementation.
# --------------------------------------------------------------------
def compute_sigma_estimate(rgb: torch.Tensor, sigma_r: float, sigma_s: float) -> torch.Tensor:
    """rgb: [..., 3, H, W]; returns same shape."""
    return torch.sqrt(sigma_r ** 2 + (sigma_s ** 2) * rgb.clamp(min=0.0))


# --------------------------------------------------------------------
# Block 2: Dataset class
# --------------------------------------------------------------------
class BurstDataset(Dataset):
    def __init__(self, npz_path: str, index_range: range):
        # ------------------------------------------------------------------
        # Block 2a: load .npz and transpose to PyTorch layout
        # ------------------------------------------------------------------
        # WHY transpose:
        #   .npz stores [N, H, W, C] and [N, K, H, W, C]. PyTorch expects
        #   [N, C, H, W] and [N, K, C, H, W]. The transpose tuple maps
        #   old axes to new axes.
        #
        # WHY ascontiguousarray:
        #   np.transpose returns a view; PyTorch .view() needs contiguous
        #   memory. ascontiguousarray forces a copy.
        # ------------------------------------------------------------------
        data = np.load(npz_path)
        self.gt = np.ascontiguousarray(np.transpose(data["truth"], (0, 3, 1, 2)))
        self.burst = np.ascontiguousarray(np.transpose(data["noisy"], (0, 3, 4, 1, 2)))
        self.sig_read = data["sig_read"]
        self.sqrt_sig_shot = data["sqrt_sig_shot"]
        self.white_level = data["white_level"]

        # ------------------------------------------------------------------
        # Block 2b: store the index range for this split
        # ------------------------------------------------------------------
        # WHY a list of ints:
        #   DataLoader calls __getitem__(0..len-1). The list maps
        #   split-local indices to global .npz indices. The groupmate
        #   provides the ranges (train/val/test), we do not re-split.
        # ------------------------------------------------------------------
        self.indices = list(index_range)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        # ------------------------------------------------------------------
        # Block 3a: map split-local to global index
        # ------------------------------------------------------------------
        global_idx = self.indices[i]

        # ------------------------------------------------------------------
        # Block 3b: fetch burst and clean reference
        # ------------------------------------------------------------------
        burst_rgb = torch.from_numpy(self.burst[global_idx]).float()  # [K, 3, H, W]
        gt_rgb = torch.from_numpy(self.gt[global_idx]).float()        # [3, H, W]

        # ------------------------------------------------------------------
        # Block 3c: read noise parameters
        # ------------------------------------------------------------------
        # WHY float(...):
        #   .npz may store them as float64; casting to Python float keeps
        #   compute_sigma_estimate's signature simple.
        #
        # WHY sqrt_sig_shot is squared inside compute_sigma_estimate:
        #   The .npz key is sqrt_sig_shot = sqrt(sigma_s). The formula
        #   needs sigma_s^2 = (sqrt_sig_shot)^2. This is the most common
        #   place for an exponent mistake; the comment is here to prevent it.
        # ------------------------------------------------------------------
        sigma_r = float(self.sig_read[global_idx])
        sigma_s = float(self.sqrt_sig_shot[global_idx])  # = sqrt(sigma_s)

        # ------------------------------------------------------------------
        # Block 3d: compute sigma and build 6-channel teacher input
        # ------------------------------------------------------------------
        # WHY concat on dim=1 (channels):
        #   burst_rgb [K, 3, H, W] + sigma_map [K, 3, H, W] -> [K, 6, H, W]
        #   where 0..2 = RGB, 3..5 = sigma.
        # ------------------------------------------------------------------
        sigma_map = compute_sigma_estimate(burst_rgb, sigma_r, sigma_s)
        teacher_input = torch.cat([burst_rgb, sigma_map], dim=1)

        # ------------------------------------------------------------------
        # Block 3e: reference frame rule (first frame)
        # ------------------------------------------------------------------
        # WHY burst_rgb[0]:
        #   Burstormer paper footnote: "We consider the first burst image
        #   to be the reference frame." The student must see the same
        #   frame the teacher anchors to.
        #
        # WHY not .clone():
        #   burst_rgb[0] is a view; we do not mutate burst_rgb after this,
        #   so the view is safe and cheaper.
        # ------------------------------------------------------------------
        reference_rgb = burst_rgb[0]

        # ------------------------------------------------------------------
        # Block 3f: return the contract dict
        # ------------------------------------------------------------------
        # Key names must match run_kd_smoke.py and the future training entry point.
        # ------------------------------------------------------------------
        return {
            "teacher_input": teacher_input,      # [K, 6, H, W]
            "reference_rgb": reference_rgb,      # [3, H, W]
            "synthetic_target": gt_rgb,          # [3, H, W]
        }