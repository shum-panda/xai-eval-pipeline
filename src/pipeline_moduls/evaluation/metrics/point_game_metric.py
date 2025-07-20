from typing import Tuple

import torch

from pipeline_moduls.evaluation.base.metric_base import MetricBase
from pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


@MetricRegistry.register("point_game")
class PointGameMetric(MetricBase):
    def calculate(self, heatmap: torch.Tensor, ground_truth: torch.Tensor) -> float:
        max_point = self._find_max_point(heatmap)
        return 1.0 if self._point_in_mask(max_point, ground_truth) else 0.0

    def _find_max_point(self, heatmap: torch.Tensor) -> Tuple[int, int]:
        if heatmap.ndim == 3:
            heatmap = heatmap[0]

        y, x = torch.nonzero(heatmap == heatmap.max(), as_tuple=True)
        return x[0].item(), y[0].item()

    def _point_in_mask(self, point: Tuple[int, int], mask: torch.Tensor) -> bool:
        if mask.ndim == 3:
            mask = mask[0]  # [1, H, W] → [H, W]
        elif mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        x, y = point
        if y >= mask.shape[0] or x >= mask.shape[1]:
            return False
        return mask[y, x].item() > 0

    def get_name(self) -> str:
        return "point_game"
