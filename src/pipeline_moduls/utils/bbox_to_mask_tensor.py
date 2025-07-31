import torch
from torch import Tensor


def bbox_to_mask_tensor(bbox, shape=(224, 224)) -> Tensor:
    """
    Wandelt eine einzelne Bounding Box (Tensor [1, 4]) in eine Binärmaske [1, H, W] um.
    """
    mask = torch.zeros((1, *shape), dtype=torch.float32, device=bbox.device)
    x1, y1, x2, y2 = bbox[0].int()  # [1, 4] → [4]
    mask[0, y1:y2, x1:x2] = 1.0
    return mask
