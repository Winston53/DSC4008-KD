"""
Synthetic smoke dataset. NOT for real training.

Returns per-sample (before DataLoader batching):
    teacher_input:     [8, 6, H, W]   -> for Burstormer teacher
    reference_rgb:     [3, H, W]      -> for NAFNet/demo student
    synthetic_target:  [3, H, W]      -> for supervised loss (toy GT)

After DataLoader with batch_size=1:
    teacher_input:     [1, 8, 6, H, W]
    reference_rgb:     [1, 3, H, W]
    synthetic_target:  [1, 3, H, W]

Why these keys:
    - teacher_input's 6 channels = 3 RGB + 3 sigma_estimate placeholder.
    - reference_rgb is EXACTLY teacher_input[0, :3] (first frame RGB).
      This enforces the "reference frame = first frame" rule at the dataset level.
    - synthetic_target is a toy clean image; it is not real GT.

TODO (real data):
    - Replace sigma_estimate placeholder with real computation:
        sigma = sqrt(sig_read**2 + sig_shot**2 * clean_rgb)
      (per-frame, per-pixel, 3 channels matching RGB)
    - Replace synthetic_target with the real clean reference for that burst.
    - Load real bursts from .npz or a synthesis pipeline (see Burstormer
      color_denoise_test_set.py for the expected format).
"""

import torch
from torch.utils.data import Dataset


class SyntheticBurstSmokeDataset(Dataset):
    def __init__(self, length=1, burst_size=8, height=128, width=128, seed=13):
        self.length = length
        self.burst_size = burst_size
        self.height = height
        self.width = width
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        K, H, W = self.burst_size, self.height, self.width

        # 1) Synthetic RGB burst: [K, 3, H, W], values in [0, 1]
        burst_rgb = torch.rand(K, 3, H, W, generator=self.generator)

        # 2) Sigma estimate PLACEHOLDER: [K, 3, H, W]
        # TODO: replace with sqrt(sig_read^2 + sig_shot^2 * clean_rgb)
        # For smoke test, just use a small constant so teacher forward is stable.
        sigma_estimate = torch.full((K, 3, H, W), 0.01)

        # 3) Teacher input: [K, 6, H, W] = concat RGB + sigma
        teacher_input = torch.cat([burst_rgb, sigma_estimate], dim=1)

        # 4) Reference RGB = first frame RGB: [3, H, W]
        # This is the "reference frame = first frame" rule, enforced here.
        reference_rgb = burst_rgb[0]

        # 5) Toy clean target: [3, H, W]
        # TODO: replace with real clean reference when using real data.
        synthetic_target = torch.rand(3, H, W, generator=self.generator)

        return {
            "teacher_input": teacher_input,
            "reference_rgb": reference_rgb,
            "synthetic_target": synthetic_target,
        }