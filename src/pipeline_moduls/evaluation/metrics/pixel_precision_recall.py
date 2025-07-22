from typing import Dict

import torch

from src.pipeline_moduls.evaluation.base.metric_base import MetricBase
from src.pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


@MetricRegistry.register("pixel_precision_recall")
class PixelPrecisionRecall(MetricBase):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def calculate(
        self, heatmap: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        pred_bin = (heatmap >= self.threshold).float()
        gt_bin = (ground_truth >= 0.5).float()  # Falls nicht exakt binär

        tp = (pred_bin * gt_bin).sum().item()
        fp = (pred_bin * (1 - gt_bin)).sum().item()
        fn = ((1 - pred_bin) * gt_bin).sum().item()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        return {"precision": precision, "recall": recall}

    def get_name(self) -> str:
        return "PixelPrecisionRecall"
