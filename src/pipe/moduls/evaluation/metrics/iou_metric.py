from typing import Any

from torch import Tensor

from src.pipe.moduls.evaluation.base.metric_base import MetricBase
from src.pipe.moduls.evaluation.base.metric_registry import MetricRegistry


@MetricRegistry.register("iou")
class IoUMetric(MetricBase):
    """
    Intersection over Union (IoU) metric for evaluating binary masks.

    Compares the binarized heatmap with the ground truth mask.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the metric with threshold for binarization.

        Args:
            **kwargs: Must include 'threshold' of type float.

        Raises:
            ValueError: If 'threshold' is missing.
            TypeError: If 'threshold' is not a float.
        """
        if "threshold" not in kwargs:
            raise ValueError(
                f"Missing required argument 'threshold' in kwargs. Got: {kwargs}"
            )

        threshold = kwargs["threshold"]
        if not isinstance(threshold, float):
            raise TypeError("'threshold' must be of type float.")

        super().__init__(**kwargs)
        self.threshold: float = threshold

    def calculate(self, heatmap: Tensor, ground_truth: Tensor) -> float:
        """
        Calculate IoU between binarized heatmap and ground truth mask.

        Args:
            heatmap (Tensor): Attribution map, shape [H, W] (2D only).
            ground_truth (Tensor): Binary mask, shape [1, H, W] or [H, W].

        Returns:
            float: IoU score between 0 and 1.
        """
        # Heatmap should already be 2D from explainer
        if heatmap.ndim > 2:
            raise ValueError(
                f"Heatmap has unexpected dimensions {heatmap.shape}. "
                f"Explainers must return 2D attributions by aggregating "
                f"multi-channel outputs in their _compute_attributions method."
            )

        if ground_truth.ndim == 3:
            ground_truth = ground_truth[0]

        heatmap_bin = (heatmap >= self.threshold).float()
        ground_truth_bin = (ground_truth > 0.5).float()  # tolerate soft masks

        intersection = (heatmap_bin * ground_truth_bin).sum().item()
        union = ((heatmap_bin + ground_truth_bin) > 0).float().sum().item()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def get_name(self) -> str:
        """
        Returns the name of the metric.

        Returns:
            str: Name identifier of the metric.
        """
        return "IoU"
