"""One-step training helper used only by the synthetic smoke test."""

from typing import Dict

import torch
from torch import nn

from losses.distillation import GroundTruthLoss, OutputKDLoss


def train_smoke_step(
    teacher: nn.Module,
    student: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    lambda_kd: float,
) -> Dict[str, float]:
    if lambda_kd < 0:
        raise ValueError("lambda_kd cannot be negative")

    teacher.eval()
    teacher.requires_grad_(False)
    student.train()

    teacher_input = batch["teacher_input"]
    reference_rgb = batch["reference_rgb"]
    target = batch["synthetic_target"]

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_output = teacher(teacher_input)
    student_output = student(reference_rgb)

    if teacher_output.shape != student_output.shape:
        raise ValueError(
            "Teacher/student output shape mismatch: "
            f"{tuple(teacher_output.shape)} != {tuple(student_output.shape)}"
        )
    if target.shape != student_output.shape:
        raise ValueError(
            "Target/student output shape mismatch: "
            f"{tuple(target.shape)} != {tuple(student_output.shape)}"
        )

    loss_gt = GroundTruthLoss()(student_output, target)
    loss_kd = OutputKDLoss()(student_output, teacher_output)
    total_loss = loss_gt + lambda_kd * loss_kd
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Smoke-test loss is not finite")

    total_loss.backward()

    if not any(parameter.grad is not None for parameter in student.parameters()):
        raise RuntimeError("Student received no gradients")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("Frozen teacher unexpectedly received gradients")

    before_update = [
        parameter.detach().clone()
        for parameter in student.parameters()
        if parameter.requires_grad
    ]
    optimizer.step()
    changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            before_update,
            (parameter for parameter in student.parameters() if parameter.requires_grad),
        )
    )
    if not changed:
        raise RuntimeError("Student parameters did not change after optimizer.step()")

    return {
        "loss_gt": float(loss_gt.detach().item()),
        "loss_kd": float(loss_kd.detach().item()),
        "loss_total": float(total_loss.detach().item()),
    }
