from dataclasses import dataclass
from pathlib import Path

import torch

@dataclass
class ImageNetSample:
    image_name: str
    image_path: Path
    image_tensor:torch.tensor
    label: int
    label_tensor: torch.tensor
    bbox_path: Path
    bbox_tensor: torch.tensor

