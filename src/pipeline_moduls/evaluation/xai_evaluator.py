import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pipeline_moduls.utils.bbox_to_mask_tensor import bbox_to_mask_tensor
from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.pipeline_moduls.evaluation.base.metric_calculator import MetricCalculator
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults
from utils.with_cuda_cleanup import with_cuda_cleanup


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
        """
        Aggregate evaluation metrics across samples to create an EvaluationSummary.

        This method computes summary statistics such as average metric values,
        prediction accuracy, and average processing time, based on the collected
        XAIExplanationResult and MetricResults.

        Args:
            results (List[XAIExplanationResult]): List of processed XAI results.
            metrics_list (List[MetricResults]): List of computed metrics per sample.
            correct_predictions (int): Number of correct model predictions.
            total_processing_time (float): Total time spent on explanation processing (in seconds).

        Returns:
            EvaluationSummary: A summary object containing aggregated metrics and metadata.
        """
        logger = self._logger
        logger.info("Starting metric aggregation...")

        explainer_name = results[0].explainer_name if results else "unknown"
        model_name = results[0].model_name if results else "unknown"
        logger.debug(f"Explainer: {explainer_name}, Model: {model_name}")
        logger.debug(f"{len(metrics_list)} samples contain metric values")

        metric_averages: Dict[str, float] = {}
        if metrics_list:
            metric_keys = list(metrics_list[0].values.keys())
            logger.debug(f"Detected metric keys: {metric_keys}")

            for key in metric_keys:
                try:
                    values = [m.values[key] for m in metrics_list if key in m.values]
                    aggregated = self._aggregate_metric(key, values)
                    metric_averages.update(aggregated)
                except Exception as e:
                    logger.warning(f"Failed to aggregate metric '{key}': {e}")
                    raise

        prediction_accuracy = correct_predictions / len(results) if results else 0.0
        average_processing_time = (
            total_processing_time / len(results) if results else 0.0
        )
        logger.debug(f"Prediction accuracy: {prediction_accuracy:.4f}")
        logger.debug(f"Average processing time: {average_processing_time:.4f}s")

        summary = EvaluationSummary(
            explainer_name=explainer_name,
            model_name=model_name,
            total_samples=len(results),
            samples_with_bbox=len(metrics_list),
            prediction_accuracy=prediction_accuracy,
            correct_predictions=correct_predictions,
            average_processing_time=average_processing_time,
            total_processing_time=total_processing_time,
            evaluation_timestamp=datetime.now().isoformat(),
            metric_averages=metric_averages,
        )

        logger.info("EvaluationSummary successfully created.")
        return summary

    def _flatten_and_average_nested_metric(
        self, key: str, values: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute average values for nested submetrics.

        This function takes a list of dictionaries representing submetrics (e.g.,
        precision, recall), and returns a dictionary with averaged values using a
        prefixed metric name.

        Args:
            key (str): The name of the parent metric (e.g., "precision_recall").
            values (List[Dict[str, float]]): A list of submetric dictionaries.

        Returns:
            Dict[str, float]: A dictionary with averaged submetrics in the form
                {"average_<key>_<sub_key>": value}.
        """
        result: Dict[str, float] = {}
        for sub_key in values[0]:
            sub_vals = [v[sub_key] for v in values if sub_key in v]
            if sub_vals:
                mean_val = float(np.mean(sub_vals))
                result[f"average_{key}_{sub_key}"] = mean_val
        return result

    def _aggregate_metric(self, key: str, values: List[Any]) -> Dict[str, float]:
        """
        Aggregate a single metric based on its type (float or nested dict).

        Args:
            logger (Any): Logger instance used for logging.
            key (str): Name of the metric to aggregate.
            values (List[Any]): List of metric values across samples. Can be float or
            nested dict.

        Returns:
            Dict[str, float]: Dictionary of averaged metrics (possibly flattened).
        """
        logger = self._logger
        result: Dict[str, float] = {}
        if not values:
            logger.warning(f"No values found for metric '{key}'")
            return result

        if isinstance(values[0], dict):
            logger.debug(
                f"Metric '{key}' contains submetrics:" f" {list(values[0].keys())}"
            )
            try:
                result.update(self._flatten_and_average_nested_metric(key, values))
            except Exception as e:
                logger.warning(f"Failed to aggregate submetrics of '{key}': {e}")
                raise
        elif isinstance(values[0], (int, float)):
            mean_val = float(np.mean(values))
            result[f"average_{key}"] = mean_val
            logger.debug(f"Aggregated: {key} -> {mean_val:.4f}")
        else:
            logger.warning(
                f"Unsupported data type for metric '{key}': {type(values[0])}"
            )
        return result

    def _log_summary(self, summary: EvaluationSummary) -> None:
        """Logge dynamische Evaluation Summary"""
        log = self._logger.info  # kürzer, aber optional

        log(f"Evaluation Summary für {summary.explainer_name}:")
        log(f"  Model:                 {summary.model_name}")
        log(f"  Total Samples:         {summary.total_samples}")
        log(f"  Samples with BBox:     {summary.samples_with_bbox}")
        log(f"  Prediction Accuracy:   {summary.prediction_accuracy:.3f}")
        log(f"  Correct Predictions:   {summary.correct_predictions}")
        log(f"  Avg Processing Time:   {summary.average_processing_time:.3f}s")
        log(f"  Total Processing Time: {summary.total_processing_time:.3f}s")
        log(f"  Evaluation Time:       {summary.evaluation_timestamp}")

        if summary.metric_averages:
            log("  Dynamische XAI-Metriken:")
            for key, value in summary.metric_averages.items():
                log(f"    {key:20}: {value:.4f}")
        else:
            log("  Keine XAI-Metriken berechnet.")
