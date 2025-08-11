import torch
from torch import Tensor


def bbox_to_mask_tensor(bbox, shape=(224, 224)) -> Tensor:
    """
    Converts bounding box tensor to binary mask, supporting multiple boxes.

    Args:
        bbox: Tensor of shape [N, 4] where N is number of bounding boxes
        shape: Target mask shape (height, width)

    Returns:
        Binary mask tensor [1, H, W] with all bounding boxes marked as 1.0
    """
    if bbox is None or bbox.numel() == 0:
        return torch.zeros((1, *shape), dtype=torch.float32)

    mask = torch.zeros((1, *shape), dtype=torch.float32, device=bbox.device)

    # Handle both single box [1, 4] and multiple boxes [N, 4]
    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)  # [4] -> [1, 4]

    for i in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[i].int()
        # Clamp coordinates to image bounds
        x1 = max(0, min(shape[1], x1))
        x2 = max(0, min(shape[1], x2))
        y1 = max(0, min(shape[0], y1))
        y2 = max(0, min(shape[0], y2))

        if x2 > x1 and y2 > y1:
            mask[0, y1:y2, x1:x2] = 1.0

    return mask
