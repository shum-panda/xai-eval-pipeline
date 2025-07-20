import logging
from typing import Any, Dict, List

import torch

from pipeline_moduls.evaluation.base.metric_base import MetricBase
from pipeline_moduls.evaluation.base.metric_registry import MetricRegistry


class MetricCalculator:
    def __init__(self, metric_names: List[str], metric_kwargs: Dict[str, Dict] = None):
        """
        Lädt und initialisiert alle gewünschten Metriken über ihre Namen.

        Args:
            metric_names: Liste der Metrik-Namen (Registrierungsschlüssel)
            metric_kwargs: Optional; Dict mit kwargs je Metrik-Name, z.B.
                {
                    "iou": {"threshold": 0.5},
                    "pixel_precision_recall": {"threshold": 0.7}
                }
        """
        self._logger = logging.getLogger(__name__)
        self.metrics: List[MetricBase] = []
        metric_kwargs = metric_kwargs or {}

        for name in metric_names:
            metric_cls = MetricRegistry.get_metric_cls(name)
            if metric_cls is None:
                raise ValueError(f"Metric '{name}' is not registered.")
            kwargs = metric_kwargs.get(name, {})
            self.metrics.append(metric_cls(**kwargs))

    def evaluate(
        self, heatmap: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Berechnet alle Metriken auf einem Heatmap/Ground Truth Paar.

        Returns:
            Dict mit Metrik-Namen als Keys und Ergebnis (float oder dict) als Value.
        """
        results = {}
        self._logger.info(f"calulating:{self.metrics}")
        for metric in self.metrics:
            result = metric.calculate(heatmap, ground_truth)
            results[metric.get_name()] = result

        self._logger.info(f"results: {results}")
        return results

    def evaluate_batch(
        self, heatmaps: torch.Tensor, ground_truths: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Batch-Auswertung: Für jede Heatmap/Ground Truth Paar in Batch berechnen.

        Args:
            heatmaps: Tensor der Form [B, ...]
            ground_truths: Tensor der Form [B, ...]

        Returns:
            Liste von Dicts mit Metrikergebnissen je Sample.
        """
        batch_results = []
        for i in range(heatmaps.size(0)):
            res = self.evaluate(heatmaps[i], ground_truths[i])
            batch_results.append(res)
        return batch_results
