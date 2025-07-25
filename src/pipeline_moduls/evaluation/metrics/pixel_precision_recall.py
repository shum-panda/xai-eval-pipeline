from typing import Any, Dict

from torch import Tensor

from src.pipeline_moduls.evaluation.base.metric_base import MetricBase
from src.pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


@MetricRegistry.register("pixel_precision_recall")
class PixelPrecisionRecall(MetricBase):
    """
    Computes pixel-wise precision and recall between a binarized heatmap and ground
    truth.

    This metric is useful for evaluating the quality of saliency maps or segmentation
    masks.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the metric with a threshold for binarization.

        Args:
            **kwargs: Must contain the key 'threshold' of type float.

        Raises:
            ValueError: If 'threshold' is missing.
            TypeError: If 'threshold' is not a float.
        """
        super().__init__(**kwargs)

        if "threshold" not in kwargs:
            raise ValueError("Missing required argument 'threshold' in metric_kwargs.")

        threshold = kwargs["threshold"]
        if not isinstance(threshold, float):
            raise TypeError("'threshold' must be of type float.")

        self.threshold: float = threshold

    def calculate(self, heatmap: Tensor, ground_truth: Tensor) -> Dict[str, float]:
        """
        Calculate pixel-wise precision and recall.

        Args:
            heatmap (Tensor): Attribution map, expected shape [1, H, W] or [H, W].
            ground_truth (Tensor): Binary mask or soft mask, shape [1, H, W] or [H, W].

        Returns:
            Dict[str, float]: Dictionary with 'precision' and 'recall'.
        """
        if heatmap.ndim == 3:
            heatmap = heatmap[0]
        if ground_truth.ndim == 3:
            ground_truth = ground_truth[0]

        pred_bin = (heatmap >= self.threshold).float()
        gt_bin = (ground_truth >= 0.5).float()  # Tolerates soft masks

        tp = (pred_bin * gt_bin).sum().item()
        fp = (pred_bin * (1 - gt_bin)).sum().item()
        fn = ((1 - pred_bin) * gt_bin).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {"precision": precision, "recall": recall}

    def get_name(self) -> str:
        """
        Returns the name of the metric.

        Returns:
            str: Name of the metric.
        """
        return "PixelPrecisionRecall"
