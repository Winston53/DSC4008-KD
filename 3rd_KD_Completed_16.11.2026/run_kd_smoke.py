"""Run one KD plumbing step with selected model factories; no quality evaluation."""

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.synthetic_smoke_dataset import SyntheticBurstSmokeDataset
from utils.smoke_training import train_smoke_step


def load_model_factory(spec: str, device: torch.device) -> nn.Module:
    """Load a user-provided `module:factory` returning a ready-to-run model."""
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError(
            f"Invalid factory '{spec}'. Use the form 'python_module:factory_name'."
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    model = factory(device)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{spec} must return a torch.nn.Module")
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one synthetic-data KD step with the selected teacher/student factories."
        )
    )
    parser.add_argument("--teacher-factory", help="Import path: module:factory")
    parser.add_argument("--student-factory", help="Import path: module:factory")
    parser.add_argument(
        "--model-path",
        help="Optional directory containing model adapters and/or Burstormer source.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-kd", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "Use --batch-size 1 for the current Burstormer forward implementation."
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.burst_size != 8:
        raise ValueError(
            "The local Burstormer color model is configured for an 8-frame burst."
        )

    if not args.teacher_factory or not args.student_factory:
        raise ValueError(
            "Provide both --teacher-factory and --student-factory."
        )

    if args.model_path:
        model_path = str(Path(args.model_path).resolve())
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        if model_path in sys.path:
            sys.path.remove(model_path)
        sys.path.insert(0, model_path)

    device = torch.device(args.device)
    dataset = SyntheticBurstSmokeDataset(
        length=1,
        burst_size=args.burst_size,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    teacher = load_model_factory(args.teacher_factory, device)
    student = load_model_factory(args.student_factory, device)

    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The student factory returned no trainable parameters")

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    batch = next(iter(loader))
    batch = {key: value.to(device) for key, value in batch.items()}

    losses = train_smoke_step(
        teacher=teacher,
        student=student,
        batch=batch,
        optimizer=optimizer,
        lambda_kd=args.lambda_kd,
    )

    print("Synthetic-data KD smoke test passed.")
    print(f"teacher_factory={args.teacher_factory}")
    print(f"student_factory={args.student_factory}")
    print("This is an integration check, not a fair comparison or quality result.")
    print(f"device={device}")
    print(f"teacher_input_shape={tuple(batch['teacher_input'].shape)}")
    print(f"reference_rgb_shape={tuple(batch['reference_rgb'].shape)}")
    print(f"synthetic_target_shape={tuple(batch['synthetic_target'].shape)}")
    print(f"loss_gt={losses['loss_gt']:.6f}")
    print(f"loss_kd={losses['loss_kd']:.6f}")
    print(f"loss_total={losses['loss_total']:.6f}")


if __name__ == "__main__":
    main()
