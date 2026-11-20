"""Download an Open Images V6 training subset with FiftyOne.

Downloaded files are stored under ``data/raw/open-images-v6/train/data``.
This only downloads clean source images; synthetic bursts and noise are
generated on demand by ``NAFNetKDBurstDataset``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = Path(__file__).resolve().parent / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a reproducible Open Images V6 train subset with FiftyOne."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of source images to download (default: 10 for a small test).",
    )
    parser.add_argument("--seed", type=int, default=4008)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dataset-name",
        help="Optional FiftyOne dataset name; defaults to a name based on max-samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive.")

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as error:
        raise ImportError(
            "Open Images downloading requires FiftyOne. Install the project "
            "requirements before running this script."
        ) from error

    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(_DATA_ROOT)
    dataset_name = args.dataset_name or f"oi_v6_train_{args.max_samples}"

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="train",
        dataset_name=dataset_name,
        label_types=[],
        max_samples=args.max_samples,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    image_dir = _DATA_ROOT / "open-images-v6" / "train" / "data"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            "FiftyOne completed loading but the expected Open Images directory "
            f"was not found: {image_dir}. First dataset filepath: "
            f"{dataset.first().filepath if len(dataset) else '<empty dataset>'}"
        )

    candidates = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    from PIL import Image

    usable_count = 0
    for path in candidates:
        try:
            with Image.open(path) as image:
                if min(image.size) >= 640:
                    usable_count += 1
        except (OSError, ValueError):
            continue

    print(f"FiftyOne dataset: {dataset_name}")
    print(f"FiftyOne records: {len(dataset)}")
    print(f"Image directory: {image_dir}")
    print(f"Image files found: {len(candidates)}")
    print(f"Images usable by the KD burst generator (>=640x640): {usable_count}")
    if usable_count < args.max_samples:
        print(
            "Note: some downloaded images are smaller than 640x640 or unreadable; "
            "the training dataset will use fewer samples than max-samples."
        )


if __name__ == "__main__":
    main()
