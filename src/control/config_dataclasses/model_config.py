from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    weights_path: Optional[str] = None
