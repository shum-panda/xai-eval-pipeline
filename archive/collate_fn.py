from typing import List, Tuple

from sympy.printing.pytorch import torch


def train_collate_fn(
    batch: List[ImageNetSample],
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Custom Collate Function für variable Anzahl von Bounding Boxes.

    Args:
        batch: Liste von (image, label, boxes) Tupeln

    Returns:
        images: Tensor [B, 3, H, W]
        labels: Tensor [B]
        boxes: Liste von Tensoren [N_i, 4]
    """
    images = torch.stack([sample.image_tensor for sample in batch])
    labels = torch.stack([sample.label_tensor for sample in batch])
    boxes = [sample.bbox_tensor for sample in batch]
    return images, labels, boxes
