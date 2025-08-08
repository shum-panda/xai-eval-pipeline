from typing import List

import torch

from src.pipe.moduls.data.dataclass.image_net_sample import ImageNetSample
from src.pipe.moduls.data.dataclass.xai_input_batch import XAIInputBatch


def explain_collate_fn(batch: List[ImageNetSample]) -> XAIInputBatch:
    """
    Custom collate function for batching ImageNetSample dataclasses.

    Args:
        batch (List[ImageNetSample]): List of ImageNetSample instances.

    Returns:
        XAIInputBatch: A structured batch containing all inputs needed for explanation,
        including images, labels, bounding boxes, paths, and more.
    """
    return XAIInputBatch(
        images_tensor=torch.stack([s.image_tensor for s in batch]),
        labels_tensor=torch.stack([s.label_tensor for s in batch]),
        boxes_list=[s.bbox_tensor for s in batch],
        image_paths=[s.image_path for s in batch],
        image_names=[s.image_path.name for s in batch],
        bbox_paths=[s.bbox_path for s in batch],
        labels_int=[s.label for s in batch],
    )
