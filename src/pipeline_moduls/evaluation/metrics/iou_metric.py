import torch

from pipeline_moduls.evaluation.base.metric_base import MetricBase
from pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


@MetricRegistry.register("iou")
class IoUMetric(MetricBase):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def calculate(self, heatmap: torch.Tensor, ground_truth: torch.Tensor) -> float:
        heatmap_bin = (heatmap >= self.threshold).float()
        intersection = (heatmap_bin * ground_truth).sum().item()
        union = ((heatmap_bin + ground_truth) > 0).float().sum().item()
        if union == 0:
            return 0.0
        return intersection / union

    def get_name(self) -> str:
        return "IoU"
