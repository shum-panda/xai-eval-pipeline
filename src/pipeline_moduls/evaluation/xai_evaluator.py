import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.control.utils.with_cuda_cleanup import with_cuda_cleanup
from src.pipeline_moduls.evaluation.base.metric_calculator import MetricCalculator
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults


def bbox_to_mask_tensor(bbox, shape=(224, 224)) -> Tensor:
    """
    Wandelt eine einzelne Bounding Box (Tensor [1, 4]) in eine Binärmaske [1, H, W] um.
    """
    mask = torch.zeros((1, *shape), dtype=torch.float32, device=bbox.device)
    x1, y1, x2, y2 = bbox[0].int()  # [1, 4] → [4]
    mask[0, y1:y2, x1:x2] = 1.0
    return mask


class XAIEvaluator:
    """
    Evaluiert XAI Ergebnisse mit verschiedenen Metriken
    """

    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._logger = logging.getLogger(__name__)
        if not metric_names:
            metric_names = ["iou", "pixel_precision_recall", "point_game"]
        if not metric_kwargs:
            self._logger.debug("Metric_kwargs is None")
            metric_kwargs = {}
        self._logger.debug(f"metric kwargs: {metric_kwargs}")
        self._logger.info("XAI Evaluator initialisiert")
        self.metric_calculator = MetricCalculator(metric_names, metric_kwargs)

    @with_cuda_cleanup
    def evaluate_single_result(
        self, result: XAIExplanationResult
    ) -> Optional[MetricResults]:
        """
        Evaluiere ein einzelnes XAI Ergebnis mit dynamischen Metriken.
        """
        if not result.has_bbox:
            return None

        bbox_mask = bbox_to_mask_tensor(result.bbox)
        if bbox_mask is None or torch.sum(bbox_mask) == 0:
            self._logger.debug(f"failed to mask: {bbox_mask}")
            return None

        attribution = result.attribution

        metric_values = self.metric_calculator.evaluate(
            heatmap=attribution, ground_truth=bbox_mask
        )
        self._logger.info(f"metric value{metric_values}")
        self._logger.debug(f"metric value{metric_values}")
        return MetricResults(values=metric_values)

    def evaluate_batch_results(
        self, results: Union[List[XAIExplanationResult], pd.DataFrame]
    ) -> EvaluationSummary:
        """
        Evaluiere eine Liste oder DataFrame von XAI Ergebnissen

        Args:
            results: Liste von XAIExplanationResult ODER DataFrame

        Returns:
            EvaluationSummary
        """
        if isinstance(results, pd.DataFrame):
            if results.empty:
                raise ValueError("Leerer DataFrame übergeben")
            # Konvertiere DataFrame → List[XAIExplanationResult]
            results = [
                XAIExplanationResult.from_dict(row.to_dict())
                for _, row in results.iterrows()
            ]

        if not results:
            raise ValueError("Keine Ergebnisse zum Evaluieren")

        self._logger.info(f"Evaluiere {len(results)} Ergebnisse...")

        metrics_list = []
        correct_predictions = 0
        total_processing_time = 0.0

        for result in tqdm(results):
            # Prediction Accuracy
            if result.prediction_correct is not None and result.prediction_correct:
                correct_predictions += 1

            total_processing_time += result.processing_time

            # XAI Metriken
            metrics = self.evaluate_single_result(result)
            if metrics:
                self._logger.info(f"append Metics {metrics} to list")
                metrics_list.append(metrics)

        summary = self._aggregate_metrics(
            results=results,
            metrics_list=metrics_list,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

        self._log_summary(summary)
        return summary

    def create_summary_from_individual_metrics(
        self,
        results: List[XAIExplanationResult],
        individual_metrics: List[MetricResults],
        correct_predictions: int,
        total_processing_time: float,
    ) -> EvaluationSummary:
        """
        Öffentliche Methode um Summary aus bereits berechneten individuellen Metriken
        zu erstellen
        """
        self._logger.info("Creating summary from pre-calculated individual metrics...")

        return self._aggregate_metrics(
            results=results,
            metrics_list=individual_metrics,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

    def _aggregate_metrics(
        self,
        results: List[XAIExplanationResult],
        metrics_list: List[MetricResults],
        correct_predictions: int,
        total_processing_time: float,
    ) -> EvaluationSummary:
        """Aggregiere Metriken zu dynamischer Summary"""

        logger = self._logger
        logger.info("Beginne Aggregation der Metriken...")

        explainer_name = results[0].explainer_name if results else "unknown"
        model_name = results[0].model_name if results else "unknown"
        logger.info(f"Explainer: {explainer_name}, Modell: {model_name}")
        logger.info(f"{len(metrics_list)} Samples mit Metriken vorhanden")

        metric_averages = {}
        if metrics_list:
            logger.debug(f"metric liste:{metrics_list}")
            metric_keys = list(metrics_list[0].values.keys())
            logger.info(f"Gefundene Metrik-Schlüssel: {metric_keys}")

            for key in metric_keys:
                try:
                    # KORREKTUR: m.values[key] statt m[key]
                    values = [m.values[key] for m in metrics_list if key in m.values]
                    logger.info(f"Bearbeite Metrik '{key}' mit {len(values)} Werten")

                    if not values:
                        logger.warning(f"Keine Werte für Metrik '{key}' gefunden")
                        continue

                    # Check for nested dicts
                    if isinstance(values[0], dict):
                        logger.info(
                            f"Metrik '{key}' enthält verschachtelte Werte: "
                            f"{list(values[0].keys())}"
                        )
                        for sub_key in values[0]:
                            try:
                                sub_vals = [v[sub_key] for v in values if sub_key in v]
                                if sub_vals:
                                    mean_val = float(np.mean(sub_vals))
                                    metric_averages[f"average_{key}_{sub_key}"] = (
                                        mean_val
                                    )
                                    logger.info(
                                        f"Aggregiert: {key}.{sub_key} -> {mean_val:.4f}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Fehler bei Sub-Metrik {key}.{sub_key}: {e}"
                                )
                    else:
                        mean_val = float(np.mean(values))
                        metric_averages[f"average_{key}"] = mean_val
                        logger.info(f"Aggregiert: {key} -> {mean_val:.4f}")

                except Exception as e:
                    logger.warning(f"Fehler bei Aggregation von {key}: {e}")

        prediction_accuracy = correct_predictions / len(results) if results else 0.0
        average_processing_time = (
            total_processing_time / len(results) if results else 0.0
        )
        logger.info(f"Prediction Accuracy: {prediction_accuracy:.4f}")
        logger.info(f"Avg Processing Time: {average_processing_time:.4f}s")

        summary = EvaluationSummary(
            explainer_name=explainer_name,
            model_name=model_name,
            total_samples=len(results),
            samples_with_bbox=len(metrics_list),
            prediction_accuracy=float(prediction_accuracy),
            correct_predictions=int(correct_predictions),
            average_processing_time=float(average_processing_time),
            total_processing_time=float(total_processing_time),
            evaluation_timestamp=datetime.now().isoformat(),
            metric_averages=metric_averages,
        )

        logger.info("EvaluationSummary erfolgreich erstellt.")
        return summary

    def _log_summary(self, summary: EvaluationSummary) -> None:
        """Logge dynamische Evaluation Summary"""
        self._logger.info(f"Evaluation Summary für {summary.explainer_name}:")
        self._logger.info(f"  Model: {summary.model_name}")
        self._logger.info(f"  Total Samples: {summary.total_samples}")
        self._logger.info(f"  Samples with BBox: {summary.samples_with_bbox}")
        self._logger.info(f"  Prediction Accuracy: {summary.prediction_accuracy:.3f}")
        self._logger.info(f"  Correct Predictions: {summary.correct_predictions}")
        self._logger.info(
            f"  Average Processing Time: {summary.average_processing_time:.3f}s"
        )
        self._logger.info(
            f"  Total Processing Time: {summary.total_processing_time:.3f}s"
        )
        self._logger.info(f"  Evaluation Time: {summary.evaluation_timestamp}")

        if summary.metric_averages:
            self._logger.info("  Dynamische XAI-Metriken:")
            for key, value in summary.metric_averages.items():
                self._logger.info(f"    {key}: {value:.4f}")
        else:
            self._logger.info("  Keine XAI-Metriken berechnet.")
