"""Train the NAFNet student from clean targets and frozen Burstormer outputs."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.nafnet_kd_dataset import NAFNetKDBurstDataset
from losses.distillation import GroundTruthLoss, OutputKDLoss


_PROJECT_ROOT = Path(__file__).resolve().parent


def _load_factory(spec: str, **kwargs: Any) -> nn.Module:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError(f"Invalid factory '{spec}'; use 'module:factory_name'.")
    factory = getattr(importlib.import_module(module_name), factory_name)
    model = factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Factory '{spec}' must return torch.nn.Module.")
    return model


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _validate_outputs(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if student_output.shape != teacher_output.shape or student_output.shape != target.shape:
        raise ValueError(
            "Teacher, student, and target outputs must have identical shapes; "
            f"got {tuple(teacher_output.shape)}, {tuple(student_output.shape)}, "
            f"and {tuple(target.shape)}."
        )


def _run_epoch(
    loader: DataLoader,
    teacher: nn.Module,
    student: nn.Module,
    device: torch.device,
    lambda_kd: float,
    ground_truth_loss: nn.Module,
    distillation_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    student.train(is_training)
    teacher.eval()
    totals = {"loss_gt": 0.0, "loss_kd": 0.0, "loss_total": 0.0}

    for batch in loader:
        batch = _move_batch(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            student_output = student(batch["reference_rgb"])
            teacher_output = teacher(batch["teacher_input"])
            target = batch["synthetic_target"]
            _validate_outputs(student_output, teacher_output, target)

            loss_gt = ground_truth_loss(student_output, target)
            loss_kd = distillation_loss(student_output, teacher_output.detach())
            loss_total = loss_gt + lambda_kd * loss_kd

            if not torch.isfinite(loss_total).item():
                raise RuntimeError("KD training loss became NaN or infinite.")
            if is_training:
                loss_total.backward()
                optimizer.step()

        totals["loss_gt"] += float(loss_gt.detach())
        totals["loss_kd"] += float(loss_kd.detach())
        totals["loss_total"] += float(loss_total.detach())

    if not len(loader):
        raise RuntimeError("A training or validation DataLoader is empty.")
    return {name: value / len(loader) for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an NAFNet student with output-level Burstormer distillation."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "open-images-v6" / "train" / "data",
        help="Open Images folder (defaults to the project-local download directory).",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=_PROJECT_ROOT / "weights" / "Burstormer.ckpt",
        help="Pretrained Burstormer Lightning checkpoint.",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        help="Optional supervised NAFNet checkpoint used to initialize the student.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "checkpoints_kd",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lambda-kd",
        type=float,
        default=0.1,
        help="Weight in L_total = L_GT + lambda_kd * L_KD; 0.1 is only an initial value.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=4008)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.lambda_kd < 0:
        raise ValueError("--lambda-kd must be non-negative.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(
            f"Burstormer checkpoint does not exist: {args.teacher_checkpoint}"
        )
    if args.student_checkpoint is not None and not args.student_checkpoint.is_file():
        raise FileNotFoundError(
            f"NAFNet checkpoint does not exist: {args.student_checkpoint}"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    training_data = NAFNetKDBurstDataset(args.image_dir, training=True, seed=args.seed)
    validation_data = NAFNetKDBurstDataset(args.image_dir, training=False, seed=args.seed)
    if training_data.image_paths != validation_data.image_paths:
        raise RuntimeError("Training and validation datasets discovered different images.")
    if len(training_data) < 2:
        raise RuntimeError("At least two usable images are required for a train/validation split.")

    indices = torch.randperm(
        len(training_data), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    validation_count = max(1, round(len(indices) * args.validation_fraction))
    if validation_count >= len(indices):
        validation_count = len(indices) - 1
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]

    training_loader = DataLoader(
        Subset(training_data, training_indices),
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    teacher = _load_factory(
        "models.burstormer.burstormer_teacher:create_teacher",
        device=device,
        checkpoint_path=args.teacher_checkpoint,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = _load_factory(
        "models.nafnet.nafnet_student:create_student",
        device=device,
        checkpoint_path=args.student_checkpoint,
    )
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("The NAFNet student has no trainable parameters.")

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    ground_truth_loss = GroundTruthLoss()
    distillation_loss = OutputKDLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    print(f"Device: {device}")
    print(f"Usable images: {len(training_data)}")
    print(f"Training/validation images: {len(training_indices)}/{len(validation_indices)}")
    print(f"lambda_kd: {args.lambda_kd}")

    for epoch in range(1, args.epochs + 1):
        training_losses = _run_epoch(
            training_loader,
            teacher,
            student,
            device,
            args.lambda_kd,
            ground_truth_loss,
            distillation_loss,
            optimizer,
        )
        with torch.no_grad():
            validation_losses = _run_epoch(
                validation_loader,
                teacher,
                student,
                device,
                args.lambda_kd,
                ground_truth_loss,
                distillation_loss,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train GT/KD/total={training_losses['loss_gt']:.6f}/"
            f"{training_losses['loss_kd']:.6f}/{training_losses['loss_total']:.6f} | "
            f"val GT/KD/total={validation_losses['loss_gt']:.6f}/"
            f"{validation_losses['loss_kd']:.6f}/{validation_losses['loss_total']:.6f}"
        )

        if validation_losses["loss_total"] < best_validation_loss:
            best_validation_loss = validation_losses["loss_total"]
            student_model = getattr(student, "model", student)
            checkpoint_path = args.output_dir / "nafnet_kd_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_losses": validation_losses,
                    "lambda_kd": args.lambda_kd,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "training_images": [
                        str(training_data.image_paths[index])
                        for index in training_indices
                    ],
                    "validation_images": [
                        str(validation_data.image_paths[index])
                        for index in validation_indices
                    ],
                },
                checkpoint_path,
            )
            print(f"Saved best student checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
