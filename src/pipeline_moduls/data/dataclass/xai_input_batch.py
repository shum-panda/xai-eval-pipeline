from dataclasses import dataclass
from typing import List
from pathlib import Path
import torch

@dataclass
class XAIInputBatch:
    """A structured batch format used by XAI explainer modules.

        Attributes:
            images_tensor (Tensor): Batched images, shape [B, C, H, W]
            labels_tensor (Tensor): Batched label tensors, shape [B]
            boxes_list (List[Tensor]): Bounding boxes per image
            image_paths (List[Path]): Absolute paths to input images
            image_names (List[str]): Filenames of the input images
            bbox_paths (List[Path]): Annotation file paths
            labels_int (List[int]): Raw integer class labels
    """
    images_tensor: torch.Tensor
    labels_tensor: torch.Tensor
    boxes_list: List[torch.Tensor]
    image_paths: List[Path]
    image_names: List[str]
    bbox_paths: List[Path]
    labels_int: List[int]

    @staticmethod
    def from_tuple(batch_tuple: tuple) -> "XAIInputBatch":
        if not isinstance(batch_tuple, tuple) or len(batch_tuple) != 7:
            raise TypeError("Expected 7-element tuple as batch input")
        return XAIInputBatch(*batch_tuple)

    def __post_init__(self):
        lengths = [
            len(self.boxes_list),
            len(self.image_paths),
            len(self.image_names),
            len(self.bbox_paths),
            len(self.labels_int),
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent list lengths: {lengths}")
