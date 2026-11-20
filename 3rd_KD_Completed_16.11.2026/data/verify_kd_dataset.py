"""Check that downloaded clean images produce usable KD burst samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.nafnet_kd_dataset import NAFNetKDBurstDataset


_DEFAULT_IMAGE_DIR = (
    Path(__file__).resolve().parent / "raw" / "open-images-v6" / "train" / "data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate a small number of KD samples."
    )
    parser.add_argument("--image-dir", type=Path, default=_DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4008)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")

    dataset = NAFNetKDBurstDataset(
        args.image_dir,
        training=False,
        seed=args.seed,
    )
    sample_count = min(args.num_samples, len(dataset))
    print(f"Usable source images: {len(dataset)}")
    print(f"Checking generated samples: {sample_count}")

    expected_shapes = {
        "teacher_input": (8, 6, 128, 128),
        "reference_rgb": (3, 128, 128),
        "synthetic_target": (3, 128, 128),
        "sigma_map": (8, 3, 128, 128),
    }

    for index in range(sample_count):
        sample = dataset[index]
        for key, shape in expected_shapes.items():
            value = sample[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(
                    f"Sample {index} has invalid {key}: "
                    f"{getattr(value, 'shape', type(value))}; expected {shape}."
                )
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Sample {index} contains non-finite values in {key}.")

        if not torch.equal(
            sample["teacher_input"][0, :3], sample["reference_rgb"]
        ):
            raise RuntimeError(f"Sample {index} reference is not burst frame 0.")
        if not torch.equal(sample["teacher_input"][:, 3:], sample["sigma_map"]):
            raise RuntimeError(f"Sample {index} sigma map is not aligned with teacher input.")

    print("KD Dataset generation and tensor checks passed.")


if __name__ == "__main__":
    main()
