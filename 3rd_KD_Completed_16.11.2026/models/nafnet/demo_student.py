"""
Demo student for KD smoke test. NOT the real NAFNet.

Purpose:
    - Satisfy the student contract: [1, 3, H, W] -> [1, 3, H, W], trainable.
    - Let run_kd_smoke.py verify the KD plumbing (loss, backward, optimizer step)
      before the real NAFNet is ready.

Contract:
    Input : [1, 3, H, W]  (reference RGB frame)
    Output: [1, 3, H, W]  (denoised image, differentiable)

Non-goals:
    - Does NOT denoise well.
    - Does NOT match NAFNet architecture.
    - Does NOT load any pretrained checkpoint.

When the real NAFNet is ready:
    - Create models/nafnet/nafnet_student.py with the same create_student() signature.
    - Point --student-factory at models.nafnet.nafnet_student:create_student.
    - This file can stay for regression testing or be deleted.
"""

import torch
import torch.nn as nn


class DemoStudent(nn.Module):
    """A tiny residual CNN. Enough to be trainable, not enough to be useful."""

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        # Simple 3-layer CNN with residual connection.
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
        )
        # Learnable residual scale: output = input + alpha * net(input).
        # This makes the initial output close to the input, which is a
        # reasonable starting point for a denoiser.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 3, H, W]
        return x + self.alpha * self.net(x)


def create_student(device):
    """Factory function called by run_kd_smoke.py via --student-factory.

    Import path: models.nafnet.demo_student:create_student
    """
    return DemoStudent().to(device)