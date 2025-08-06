from typing import Tuple

import torch
from torch import Tensor

from src.pipeline.pipeline_moduls.evaluation.base.metric_base import MetricBase
from src.pipeline.pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


@MetricRegistry.register("point_game")
class PointGameMetric(MetricBase):
    """
    Implements the 'Pointing Game' metric.

    This metric evaluates whether the maximum attribution point lies within the
    ground-truth mask.
    """

    def calculate(self, heatmap: Tensor, ground_truth: Tensor) -> float:
        """
        Compute the Pointing Game score (1.0 or 0.0).

        Args:
            heatmap (Tensor): Attribution heatmap of shape [H, W] (2D only).
            ground_truth (Tensor): Binary mask of shape [1, H, W] or [H, W].

        Returns:
            float: 1.0 if max point is inside mask, else 0.0.
        """
        max_point = self._find_max_point(heatmap)
        return 1.0 if self._point_in_mask(max_point, ground_truth) else 0.0

    def _find_max_point(self, heatmap: Tensor) -> Tuple[int, int]:
        """
        Find the (x, y) coordinate of the maximum value in the heatmap.

        Args:
            heatmap (Tensor): Tensor of shape [H, W] (2D only).

        Returns:
            Tuple[int, int]: Coordinates (x, y) of maximum value.
        """
        if heatmap.ndim != 2:
            raise ValueError(
                f"Heatmap has unexpected dimensions {heatmap.shape}. "
                f"Explainers must return 2D attributions (H, W) by aggregating "
                f"multi-channel outputs in their _compute_attributions method."
            )

        y, x = torch.nonzero(heatmap == heatmap.max(), as_tuple=True)

        # In case of multiple max points, take the first
        return int(x[0].item()), int(y[0].item())

    def _point_in_mask(self, point: Tuple[int, int], mask: Tensor) -> bool:
        """
        Check whether a given point lies within the binary mask.

        Args:
            point (Tuple[int, int]): (x, y) coordinate to check.
            mask (Tensor): Binary mask of shape [1, H, W] or [H, W].

        Returns:
            bool: True if point is inside the mask, else False.
        """
        if mask.ndim == 3:
            mask = mask[0]
        elif mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        x, y = point
        if y >= mask.shape[0] or x >= mask.shape[1]:
            return False

        return mask[y, x].item() > 0

    def get_name(self) -> str:
        """
        Returns the unique name of this metric.

        Returns:
            str: Name of the metric.
        """
        return "point_game"
