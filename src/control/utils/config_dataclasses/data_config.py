from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DataConfig:
    dataset_name: str = "imagenet_val"
    dataset_path: str = "data/extracted/validation_images"  # Path to dataset images
    annotation_path: str = "data/extracted/bounding_boxes"  # Path to annotations
    label_file: str = (
        "data/ILSVRC2012_validation_ground_truth.txt"  # Labels for evaluation
    )
    shuffle: bool = False
    resize: List[int] = field(default_factory=lambda: [224, 224])
    normalize: bool = True
    augmentation: Dict[str, bool] = field(
        default_factory=lambda: {"horizontal_flip": False, "random_crop": False}
    )
    max_samples: int = 200
    batch_size: int = 32
    max_batches: int = int(max_samples / batch_size)
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if len(self.resize) != 2:
            raise ValueError("resize must be a list of two integers")
