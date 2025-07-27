import logging
from typing import Any, Dict, List, Optional

from torch import Tensor

from src.pipeline_moduls.evaluation.base.metric_base import MetricBase
from src.pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


class MetricCalculator:
    """
    Loader and evaluator for multiple registered metrics.

    Allows initialization of multiple metrics by name and evaluation
    on single samples or batches.
    """

    def __init__(
        self,
        metric_names: List[str],
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize and instantiate metrics by their registration names.

        Args:
            metric_names: List of metric registration names.
            metric_kwargs: Optional dictionary mapping metric names to kwargs dict.
                Example:
                {
                    "iou": {"threshold": 0.5},
                    "pixel_precision_recall": {"threshold": 0.7}
                }
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self.metrics: List[MetricBase] = []
        metric_kwargs = metric_kwargs or {}

        for name in metric_names:
            metric_cls = MetricRegistry.get_metric_cls(name)
            if metric_cls is None:
                raise ValueError(f"Metric '{name}' is not registered.")
            kwargs = metric_kwargs.get(name, {})
            self.metrics.append(metric_cls(**kwargs))

    def evaluate(self, heatmap: Tensor, ground_truth: Tensor) -> Dict[str, Any]:
        """
        Evaluate all configured metrics on a single heatmap/ground truth pair.

        Args:
            heatmap: Attribution heatmap tensor.
            ground_truth: Ground truth tensor.

        Returns:
            Dictionary mapping metric names to their computed result.
        """
        results: Dict[str, Any] = {}
        self._logger.debug(
            f"Calculating metrics: {[m.get_name() for m in self.metrics]}"
        )
        for metric in self.metrics:
            result = metric.calculate(heatmap, ground_truth)
            results[metric.get_name()] = result

        self._logger.debug(f"Metric results: {results}")
        return results

    def evaluate_batch(
        self, heatmaps: Tensor, ground_truths: Tensor
    ) -> List[Dict[str, Any]]:
        """
        Evaluate metrics for each heatmap/ground truth pair in a batch.

        Args:
            heatmaps: Tensor of shape [B, ...].
            ground_truths: Tensor of shape [B, ...].

        Returns:
            List of dicts containing metric results per sample.
        """
        batch_results: List[Dict[str, Any]] = []
        for i in range(heatmaps.size(0)):
            res = self.evaluate(heatmaps[i], ground_truths[i])
            batch_results.append(res)
        return batch_results
