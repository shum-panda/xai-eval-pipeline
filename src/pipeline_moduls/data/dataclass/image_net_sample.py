from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ImageNetSample:
    image_name: str
    image_path: Path
    image_tensor: torch.tensor  # type: ignore
    label: int
    label_tensor: torch.tensor  # type: ignore
    bbox_path: Path
    bbox_tensor: torch.tensor  # type: ignore
