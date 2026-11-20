"""Pixel-space losses for the output-level KD demonstration.

Why L1 (not L2, not Softmax+KL):
    - Image denoising outputs are continuous tensors, not class probabilities.
      Softmax+KL (as in classification KD) does not apply.
    - L1 is the standard choice in image restoration benchmarks (SIDD, NAFNet)
      and is robust to outliers compared to L2.
    - L1 is also what Burstormer's official evaluation uses (via PSNR,
      which is derived from MSE, but the training loss is L1).

Why two separate loss objects (not a single combined one):
    - The training loop needs to log loss_gt and loss_kd separately to
      diagnose whether one is dominating.
    - The weight lambda_kd is applied outside these objects, in the
      training loop / config, so it can be tuned independently.
"""

from torch import Tensor
from torch import nn


class OutputKDLoss(nn.Module):
    """L1 distance between student and teacher image outputs.

    Input:
        student_output: [B, 3, H, W]  (has grad_fn, flows into student)
        teacher_output: [B, 3, H, W]  (no grad_fn, treated as constant)
    Returns:
        scalar tensor, differentiable w.r.t. student_output
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        return self.loss(student_output, teacher_output)


class GroundTruthLoss(nn.Module):
    """L1 distance between student output and the ground-truth target.

    Input:
        student_output: [B, 3, H, W]  (has grad_fn)
        target:         [B, 3, H, W]  (clean reference, no grad)
    Returns:
        scalar tensor, differentiable w.r.t. student_output
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, student_output: Tensor, target: Tensor) -> Tensor:
        return self.loss(student_output, target)