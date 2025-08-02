# streaming_evaluation.py - New streaming evaluation system

from typing import Iterator, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import logging
from tqdm import tqdm

from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import \
    EvaluationSummary
from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults
from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult


@dataclass
class StreamingMetricState:
    """
    Maintains running statistics for streaming evaluation.
    Only stores aggregated values, not individual results.
    """
    # Basic counters
    total_samples: int = 0
    correct_predictions: int = 0
    samples_with_bbox: int = 0
    samples_processed: int = 0

    # Time tracking
    total_processing_time: float = 0.0
    total_evaluation_time: float = 0.0
    evaluation_start_time: float = 0.0

    # Running metric sums for online averaging
    metric_sums: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)

    # Nested metrics (like precision_recall)
    nested_metric_sums: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nested_metric_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Metadata
    explainer_name: str = "unknown"
    model_name: str = "unknown"

    def reset(self):
        """Reset all counters for new evaluation run"""
        self.total_samples = 0
        self.correct_predictions = 0
        self.samples_with_bbox = 0
        self.samples_processed = 0
        self.total_processing_time = 0.0
        self.total_evaluation_time = 0.0
        self.metric_sums.clear()
        self.metric_counts.clear()
        self.nested_metric_sums.clear()
        self.nested_metric_counts.clear()
        self.evaluation_start_time = time.time()


class StreamingEvaluator:
    """
    Memory-efficient streaming evaluator that processes results batch-wise
    and maintains only running statistics, not individual results.
    """

    def __init__(self, base_evaluator, logger: Optional[logging.Logger] = None):
        self._base_evaluator = base_evaluator
        self._logger = logger or logging.getLogger(__name__)
        self._state = StreamingMetricState()

        # For CSV export - store only if needed
        self._store_individual = False
        self._individual_metrics = []

    def reset_for_new_run(self, store_individual: bool = False):
        """
        Reset evaluator state for new evaluation run.

        Args:
            store_individual: Whether to store individual metrics for CSV export
        """
        self._state.reset()
        self._store_individual = store_individual
        if store_individual:
            self._individual_metrics = []
        self._logger.info("Streaming evaluator reset for new run")

    def process_result_stream(
            self,
            result_stream: Iterator[XAIExplanationResult],
            batch_size: int = 50
    ) -> Iterator[Dict[str, Any]]:
        """
        Process XAI results in streaming fashion, yielding progress updates.

        Args:
            result_stream: Iterator of XAI results
            batch_size: Number of results to process before yielding progress

        Yields:
            Progress dictionaries with current statistics
        """
        batch_count = 0
        batch_results = []

        for result in result_stream:
            # Process single result
            self._process_single_result(result)
            batch_results.append(result)

            # Yield progress every batch_size results
            if len(batch_results) >= batch_size:
                batch_count += 1
                progress = self._create_progress_update(batch_count, batch_size)

                # Clear batch from memory
                del batch_results
                batch_results = []

                # Force garbage collection for large batches
                import gc
                gc.collect()

                yield progress

        # Process remaining results
        if batch_results:
            batch_count += 1
            progress = self._create_progress_update(batch_count, len(batch_results))
            yield progress

    def _process_single_result(self, result: XAIExplanationResult) -> None:
        """Process a single XAI result and update running statistics"""
        eval_start = time.time()

        # Update basic counters
        self._state.total_samples += 1
        self._state.samples_processed += 1

        # Update metadata from first result
        if self._state.total_samples == 1:
            self._state.explainer_name = result.explainer_name
            self._state.model_name = result.model_name

        # Update prediction accuracy
        if result.prediction_correct:
            self._state.correct_predictions += 1

        # Update processing time
        self._state.total_processing_time += result.processing_time

        # Evaluate XAI metrics if result has bbox
        if result.has_bbox:
            try:
                metrics = self._base_evaluator.evaluate_single_result(result)
                if metrics and metrics.values:
                    self._update_metric_statistics(metrics)
                    self._state.samples_with_bbox += 1

                    # Store individual metrics if needed for CSV
                    if self._store_individual:
                        self._individual_metrics.append(metrics)

            except Exception as e:
                self._logger.warning(f"Failed to evaluate {result.image_name}: {e}")
                if self._store_individual:
                    self._individual_metrics.append(None)
        else:
            if self._store_individual:
                self._individual_metrics.append(None)

        # Update evaluation timing
        eval_end = time.time()
        self._state.total_evaluation_time += (eval_end - eval_start)

        # Clear result attribution from memory immediately
        if hasattr(result, 'attribution'):
            del result.attribution

    def _update_metric_statistics(self, metrics: MetricResults) -> None:
        """Update running metric statistics with new metrics"""
        for metric_name, metric_value in metrics.values.items():
            if isinstance(metric_value, (int, float)):
                # Simple numeric metric - update running average
                if metric_name not in self._state.metric_sums:
                    self._state.metric_sums[metric_name] = 0.0
                    self._state.metric_counts[metric_name] = 0

                self._state.metric_sums[metric_name] += float(metric_value)
                self._state.metric_counts[metric_name] += 1

            elif isinstance(metric_value, dict):
                # Nested metric (e.g., precision_recall)
                if metric_name not in self._state.nested_metric_sums:
                    self._state.nested_metric_sums[metric_name] = {}
                    self._state.nested_metric_counts[metric_name] = {}

                for sub_key, sub_value in metric_value.items():
                    if isinstance(sub_value, (int, float)):
                        if sub_key not in self._state.nested_metric_sums[metric_name]:
                            self._state.nested_metric_sums[metric_name][sub_key] = 0.0
                            self._state.nested_metric_counts[metric_name][sub_key] = 0

                        self._state.nested_metric_sums[metric_name][sub_key] += float(
                            sub_value)
                        self._state.nested_metric_counts[metric_name][sub_key] += 1

    def _create_progress_update(self, batch_num: int, batch_size: int) -> Dict[
        str, Any]:
        """Create progress update dictionary"""
        current_averages = self.get_current_metric_averages()

        return {
            "batch_number": batch_num,
            "batch_size": batch_size,
            "total_processed": self._state.samples_processed,
            "samples_with_bbox": self._state.samples_with_bbox,
            "prediction_accuracy": self._get_current_accuracy(),
            "avg_processing_time": self._get_current_avg_processing_time(),
            "current_metric_averages": current_averages,
            "evaluation_time_elapsed": time.time() - self._state.evaluation_start_time
        }

    def get_current_metric_averages(self) -> Dict[str, float]:
        """Get current metric averages from running statistics"""
        averages = {}

        # Simple metrics
        for metric_name, total_sum in self._state.metric_sums.items():
            count = self._state.metric_counts[metric_name]
            if count > 0:
                averages[f"average_{metric_name}"] = total_sum / count

        # Nested metrics
        for metric_name, sub_metrics in self._state.nested_metric_sums.items():
            for sub_key, sub_sum in sub_metrics.items():
                count = self._state.nested_metric_counts[metric_name][sub_key]
                if count > 0:
                    averages[f"average_{metric_name}_{sub_key}"] = sub_sum / count

        return averages

    def _get_current_accuracy(self) -> float:
        """Get current prediction accuracy"""
        if self._state.total_samples == 0:
            return 0.0
        return self._state.correct_predictions / self._state.total_samples

    def _get_current_avg_processing_time(self) -> float:
        """Get current average processing time"""
        if self._state.total_samples == 0:
            return 0.0
        return self._state.total_processing_time / self._state.total_samples

    def create_final_summary(self) -> EvaluationSummary:
        """Create final evaluation summary from accumulated statistics"""
        metric_averages = self.get_current_metric_averages()

        return EvaluationSummary(
            explainer_name=self._state.explainer_name,
            model_name=self._state.model_name,
            total_samples=self._state.total_samples,
            samples_with_bbox=self._state.samples_with_bbox,
            prediction_accuracy=self._get_current_accuracy(),
            correct_predictions=self._state.correct_predictions,
            average_processing_time=self._get_current_avg_processing_time(),
            total_processing_time=self._state.total_processing_time,
            evaluation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metric_averages=metric_averages
        )

    def get_individual_metrics_for_csv(self) -> Optional[list]:
        """Return stored individual metrics for CSV export, if enabled"""
        if self._store_individual:
            return self._individual_metrics
        return None