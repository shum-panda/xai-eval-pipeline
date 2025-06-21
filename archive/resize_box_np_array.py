from typing import Tuple

import numpy as np


def resize_boxes(boxes: np.ndarray, orig_size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
    """
    Skaliert Bounding Boxes entsprechend der Bildgrößenänderung.

    Args:
        boxes: Bounding Boxes [N, 4] im Format [xmin, ymin, xmax, ymax]
        orig_size: Original Bildgröße (W, H)
        new_size: Neue Bildgröße (W, H)

    Returns:
        np.ndarray: Skalierte Bounding Boxes
    """
    if len(boxes) == 0:
        return boxes

    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    # Skalierungsfaktoren
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Skaliere Koordinaten
    boxes_scaled = boxes.copy()
    boxes_scaled[:, [0, 2]] *= scale_x  # xmin, xmax
    boxes_scaled[:, [1, 3]] *= scale_y  # ymin, ymax

    # Stelle sicher, dass Koordinaten im gültigen Bereich bleiben
    boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, new_w)
    boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, new_h)

    return boxes_scaled
