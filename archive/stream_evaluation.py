import logging
from typing import List
from dataclasses import dataclass
import time

from src.pipeline.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.pipeline.pipeline_moduls.evaluation.dataclass.evaluation_summary import EvaluationSummary
from src.pipeline.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults


@dataclass
class StreamingEvaluationState:
    """
    Maintains running statistics during streaming evaluation.
    """

    total_samples: int = 0
    correct_predictions: int = 0
    samples_with_bbox: int = 0
    total_processing_time: float = 0.0

    # Evaluation timing
    evaluation_start_time: float = 0.0
    total_evaluation_time: float = 0.0

    # Running metric sums for averaging
    total_iou: float = 0.0
    total_point_game_hits: int = 0
    total_coverage: float = 0.0
    total_precision: float = 0.0
    total_recall: float = 0.0

    # For detailed metrics
    metric_sums: dict = None
    metric_counts: dict = None

    def __post_init__(self):
        if self.metric_sums is None:
            self.metric_sums = {}
        if self.metric_counts is None:
            self.metric_counts = {}


class StreamingEvaluator:
    """
    Evaluator that processes results one-by-one in streaming fashion.
    Maintains running statistics without storing all individual results.
    """

    def __init__(self, evaluator):
        self._evaluator = evaluator
        self._logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        """Reset evaluation state for new evaluation run"""
        self._state = StreamingEvaluationState()
        self._individual_metrics = []  # Only store if needed for CSV
        # Start evaluation timing
        self._state.evaluation_start_time = time.time()
        self._logger.info("Starting streaming evaluation with timing...")

    def evaluate_single_and_update(
        self, result: XAIExplanationResult, store_individual: bool = True
    ) -> MetricResults:
        """
        Evaluate single result and update running statistics.

        Args:
            result: XAI result to evaluate
            store_individual: Whether to store individual metrics for CSV export

        Returns:
            Individual metrics for this result
        """
        # Time the evaluation of this single result
        eval_start = time.time()

        # Evaluate single result
        metrics = self._evaluator.evaluate_single_result(result)

        eval_end = time.time()
        single_eval_time = eval_end - eval_start

        # Update running statistics
        self._update_statistics(result, metrics, single_eval_time)

        # Store individual metrics if needed (for CSV export)
        if store_individual:
            self._individual_metrics.append(metrics)

        # Progress logging every 10 results
        if self._state.total_samples % 10 == 0:
            self._logger.info(
                f"Processed {self._state.total_samples} individual metrics "
                f"(eval time: {single_eval_time:.3f}s)"
            )

        # Clear attribution cache immediately after evaluation
        if hasattr(result, "clear_attribution_cache"):
            result.clear_attribution_cache()

        return metrics

    def _update_statistics(
        self, result: XAIExplanationResult, metrics: MetricResults, eval_time: float
    ):
        """Update running statistics with new result"""
        state = self._state

        # Basic counters
        state.total_samples += 1
        if result.prediction_correct:
            state.correct_predictions += 1
        if result.has_bbox:
            state.samples_with_bbox += 1
        state.total_processing_time += result.processing_time
        state.total_evaluation_time += eval_time  # Track evaluation time

        # Update metric statistics
        if metrics and metrics.values:
            for metric_name, metric_value in metrics.values.items():
                if isinstance(metric_value, (int, float)):
                    # Simple numeric metric
                    if metric_name not in state.metric_sums:
                        state.metric_sums[metric_name] = 0.0
                        state.metric_counts[metric_name] = 0

                    state.metric_sums[metric_name] += float(metric_value)
                    state.metric_counts[metric_name] += 1

                elif isinstance(metric_value, dict):
                    # Nested metric (e.g., PixelPrecisionRecall)
                    for sub_key, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            full_key = f"{metric_name}_{sub_key}"
                            if full_key not in state.metric_sums:
                                state.metric_sums[full_key] = 0.0
                                state.metric_counts[full_key] = 0

                            state.metric_sums[full_key] += float(sub_value)
                            state.metric_counts[full_key] += 1

    def get_current_summary(self) -> EvaluationSummary:
        """
        Get current evaluation summary from running statistics.
        Can be called at any time during streaming evaluation.
        """
        state = self._state

        # Calculate averages
        prediction_accuracy = (
            state.correct_predictions / state.total_samples
            if state.total_samples > 0
            else 0.0
        )

        average_processing_time = (
            state.total_processing_time / state.total_samples
            if state.total_samples > 0
            else 0.0
        )

        # Calculate evaluation timing
        current_time = time.time()
        total_elapsed_time = current_time - state.evaluation_start_time
        average_evaluation_time_per_sample = (
            state.total_evaluation_time / state.total_samples
            if state.total_samples > 0
            else 0.0
        )

        # Calculate metric averages
        metric_averages = {}
        for metric_name, total_sum in state.metric_sums.items():
            count = state.metric_counts[metric_name]
            if count > 0:
                metric_averages[f"average_{metric_name}"] = total_sum / count

        # Add timing metrics
        metric_averages["total_evaluation_time"] = state.total_evaluation_time
        metric_averages["average_evaluation_time_per_sample"] = (
            average_evaluation_time_per_sample
        )
        metric_averages["total_elapsed_time"] = total_elapsed_time

        # Create summary
        summary = EvaluationSummary(
            total_samples=state.total_samples,
            samples_with_bbox=state.samples_with_bbox,
            prediction_accuracy=prediction_accuracy,
            average_processing_time=average_processing_time,
            metric_averages=metric_averages,
            explainer_name=getattr(self, "_current_explainer_name", "unknown"),
            model_name=getattr(self, "_current_model_name", "unknown"),
        )

        return summary

    def finalize_evaluation(self) -> EvaluationSummary:
        """
        Finalize evaluation and log comprehensive timing information.
        """
        final_summary = self.get_current_summary()
        state = self._state

        # Log detailed timing information
        self._logger.info("=== EVALUATION TIMING SUMMARY ===")
        self._logger.info(f"Total samples processed: {state.total_samples}")
        self._logger.info(f"Total evaluation time: {state.total_evaluation_time:.2f}s")
        self._logger.info(
            f"Average time per sample: {state.total_evaluation_time / state.total_samples:.4f}s"
        )
        self._logger.info(
            f"Evaluation throughput: {state.total_samples / state.total_evaluation_time:.1f} samples/sec"
        )
        self._logger.info("Evaluation metrics calculation finished!")

        return final_summary

    def get_individual_metrics(self) -> List[MetricResults]:
        """Get stored individual metrics (for CSV export)"""
        return self._individual_metrics.copy()

    def clear_individual_metrics(self):
        """Clear stored individual metrics to free memory"""
        self._individual_metrics.clear()
        import gc

        gc.collect()


# Modified run_pipeline with streaming evaluation
