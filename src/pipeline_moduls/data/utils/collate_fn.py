from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor

from pipeline_moduls.data.dataclass.image_net_sample import ImageNetSample


def explain_collate_fn(
    batch: List[ImageNetSample],
) -> Tuple[
    Tensor,  # images stacked [B, C, H, W]
    Tensor,  # labels stacked [B]
    List[Tensor],  # bounding box tensors, variable length per sample
    List[Path],  # full image paths
    List[str],  # image file names (only the name)
    List[Path],  # bounding box annotation paths
    List[int],  # labels as integers
]:
    """
    Custom collate function for batching ImageNetSample dataclasses.

    Args:
        batch (List[ImageNetSample]): List of ImageNetSample instances.

    Returns:
        Tuple containing:
            - images (Tensor): Batch of images stacked into a single tensor.
            - labels_tensor (Tensor): Batch of labels stacked into a tensor.
            - boxes (List[Tensor]): List of bounding box tensors for each sample.
            - image_paths (List[Path]): List of full file paths for images.
            - image_names (List[str]): List of image file names extracted from paths.
            - bbox_paths (List[Path]): List of full file paths for bounding box annotations.
            - labels (List[int]): List of label integers.
    """
    images = torch.stack([sample.image_tensor for sample in batch])
    labels_tensor = torch.stack([sample.label_tensor for sample in batch])
    boxes = [sample.bbox_tensor for sample in batch]
    image_paths = [sample.image_path for sample in batch]
    image_names = [sample.image_path.name for sample in batch]
    bbox_paths = [sample.bbox_path for sample in batch]
    labels = [sample.label for sample in batch]

    result = (
        images,
        labels_tensor,
        boxes,
        image_paths,
        image_names,
        bbox_paths,
        labels,
    )
    return result
