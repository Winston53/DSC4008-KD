"""Generate aligned synthetic bursts for Burstormer-teacher/NAFNet-student KD.

The burst geometry and heteroscedastic noise distribution follow the
groupmate's ``burst_utils.py`` without modifying or importing that project.
Each item keeps the full burst, its first/reference frame, the clean target,
and the exact noise parameters together so the teacher noise maps correspond
to the noise actually synthesized for that item.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_BURST_SIZE = 8
_MIN_IMAGE_SIZE = 640
_REGION_SIZE = 640
_CROP_SIZE = 512
_FRAME_SIZE = 128
_DOWNSAMPLE = 4


def _generate_clean_burst(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    if min(height, width) < _MIN_IMAGE_SIZE:
        raise ValueError(
            f"Images must be at least {_MIN_IMAGE_SIZE}x{_MIN_IMAGE_SIZE}; "
            f"received {width}x{height}."
        )

    top = int(rng.integers(0, height - _REGION_SIZE + 1))
    left = int(rng.integers(0, width - _REGION_SIZE + 1))
    region = image[top : top + _REGION_SIZE, left : left + _REGION_SIZE]
    large_motion_probability = min(float(rng.poisson(1.5)) / _BURST_SIZE, 1.0)
    frames = []

    for frame_index in range(_BURST_SIZE):
        if frame_index == 0:
            dy, dx = 0, 0
        else:
            limit = 16 if rng.random() < large_motion_probability else 2
            dy, dx = rng.integers(-limit * _DOWNSAMPLE, limit * _DOWNSAMPLE + 1, size=2)

        y, x = 64 + int(dy), 64 + int(dx)
        crop = region[y : y + _CROP_SIZE, x : x + _CROP_SIZE]
        frame = crop.reshape(
            _FRAME_SIZE, _DOWNSAMPLE, _FRAME_SIZE, _DOWNSAMPLE, 3
        ).mean(axis=(1, 3))
        frames.append(frame)

    return np.stack(frames).astype(np.float32, copy=False)


def _synthesize_noisy_burst(
    clean_burst: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    white_level = float(10 ** rng.uniform(-1.0, 0.0))
    clean_linear = (white_level * clean_burst**2.2).astype(np.float32)
    sigma_read = float(10 ** rng.uniform(-3.0, -1.5))
    sigma_shot = float(10 ** rng.uniform(-2.0, -1.0))

    read_noise = sigma_read * rng.standard_normal(clean_linear.shape)
    shot_noise = (
        np.sqrt(clean_linear)
        * sigma_shot
        * rng.standard_normal(clean_linear.shape)
    )
    noisy_burst = (clean_linear + read_noise + shot_noise).astype(np.float32)
    sigma_map = np.sqrt(
        sigma_read**2 + sigma_shot**2 * np.maximum(noisy_burst, 0.0)
    ).astype(np.float32)

    return clean_linear, noisy_burst, sigma_map, white_level, sigma_read, sigma_shot


class NAFNetKDBurstDataset(Dataset):
    """Create aligned full-burst teacher inputs and single-frame student data.

    The first burst frame is the reference frame for both student input and
    ground-truth target. ``training=True`` resamples crops/noise on every
    access; validation samples are reproducible for a fixed ``seed``.

    Returned tensors are unbatched:
        teacher_input:   [8, 6, 128, 128] (RGB + per-pixel sigma map)
        reference_rgb:   [3, 128, 128]
        synthetic_target:[3, 128, 128]
        sigma_map:       [8, 3, 128, 128]
    """

    def __init__(
        self,
        image_dir: str | Path,
        training: bool = True,
        seed: int = 4008,
    ) -> None:
        self.image_dir = Path(image_dir).expanduser()
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        self.training = training
        self.seed = int(seed)
        extensions = {".jpg", ".jpeg", ".png"}
        candidates = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        self.image_paths = []
        for path in candidates:
            try:
                with Image.open(path) as image:
                    if min(image.size) >= _MIN_IMAGE_SIZE:
                        self.image_paths.append(path)
            except (OSError, ValueError):
                continue

        if not self.image_paths:
            raise ValueError(
                f"No readable images at least {_MIN_IMAGE_SIZE}x{_MIN_IMAGE_SIZE} "
                f"were found in {self.image_dir}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        sample_seed = (
            int(torch.randint(0, 2**32, ()).item())
            if self.training
            else self.seed + index
        )
        rng = np.random.default_rng(sample_seed)

        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0

        clean_burst = _generate_clean_burst(image, rng)
        (
            clean_linear,
            noisy_burst,
            sigma_map,
            white_level,
            sigma_read,
            sigma_shot,
        ) = _synthesize_noisy_burst(clean_burst, rng)

        burst_tensor = torch.from_numpy(noisy_burst).permute(0, 3, 1, 2).contiguous()
        sigma_tensor = torch.from_numpy(sigma_map).permute(0, 3, 1, 2).contiguous()
        target_tensor = torch.from_numpy(clean_linear[0]).permute(2, 0, 1).contiguous()
        teacher_input = torch.cat((burst_tensor, sigma_tensor), dim=1)

        return {
            "teacher_input": teacher_input,
            "reference_rgb": burst_tensor[0],
            "synthetic_target": target_tensor,
            "sigma_map": sigma_tensor,
            "white_level": torch.tensor(white_level, dtype=torch.float32),
            "sigma_read": torch.tensor(sigma_read, dtype=torch.float32),
            "sigma_shot": torch.tensor(sigma_shot, dtype=torch.float32),
            "sample_seed": torch.tensor(sample_seed, dtype=torch.int64),
            "image_path": str(image_path),
        }
