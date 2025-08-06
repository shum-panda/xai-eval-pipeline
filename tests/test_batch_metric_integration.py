from pathlib import Path
from typing import List, Optional

import pytest
import torch

from src.pipeline.control.utils.dataclasses.xai_explanation_result import (
    XAIExplanationResult,
)
from src.pipeline.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline.pipeline_moduls.evaluation.dataclass.metricresults import (
    MetricResults,
)
from src.pipeline.pipeline_moduls.evaluation.xai_evaluator import XAIEvaluator


class TestBatchMetricIntegration:
    """
    Integration tests for the complete batch metric evaluation pipeline.
    Tests the interaction between XAIEvaluator and the orchestrator.
    """

    def create_realistic_result(
        self,
        image_name: str,
        has_bbox: bool = True,
        bbox_coords: Optional[List[float]] = None,
        prediction_correct: bool = True,
    ) -> XAIExplanationResult:
        """Create a realistic XAI result for testing."""

        # Create realistic image and attribution tensors
        image = torch.randn(3, 224, 224)

        # Create attribution with some structure (hot spot in center)
        attribution = torch.randn(3, 224, 224)
        if has_bbox:
            # Add a hot spot in the attribution where the bbox should be
            if bbox_coords:
                x1, y1, x2, y2 = bbox_coords
            else:
                x1, y1, x2, y2 = 80, 80, 144, 144

            # Make sure coordinates are within bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(224, int(x2)), min(224, int(y2))

            # Add signal in the bbox region
            attribution[:, y1:y2, x1:x2] += 2.0

        # Create bbox tensor
        if has_bbox and bbox_coords:
            bbox = torch.tensor(bbox_coords, dtype=torch.float32)
        elif has_bbox:
            bbox = torch.tensor([80.0, 80.0, 144.0, 144.0], dtype=torch.float32)
        else:
            bbox = torch.tensor([], dtype=torch.float32)

        return XAIExplanationResult(
            image=image,
            image_path=Path(f"/test/images/{image_name}.jpg"),
            image_name=image_name,
            predicted_class=1,
            prediction_confidence=0.85
            + torch.rand(1).item() * 0.14,  # Random but high confidence
            true_label=1 if prediction_correct else 0,
            prediction_correct=prediction_correct,
            topk_predictions=[1, 2, 3, 4, 5],
            topk_confidences=[0.85, 0.08, 0.04, 0.02, 0.01],
            attribution=attribution,
            explainer_result=None,
            explainer_name="grad_cam",
            explainer_params={"layer_name": "features.29"},
            has_bbox=has_bbox,
            bbox=bbox,
            model_name="resnet50",
            processing_time=0.1
            + torch.rand(1).item() * 0.05,  # Realistic processing time
            timestamp=str(1234567890 + hash(image_name) % 10000),
        )

    @pytest.fixture
    def xai_evaluator(self):
        """Create XAI evaluator with comprehensive metrics."""
        return XAIEvaluator(
            metric_names=["iou", "point_game", "pixel_precision_recall"],
            metric_kwargs={
                "iou": {},
                "point_game": {},
                "pixel_precision_recall": {"threshold": 0.5},
            },
        )

    def test_realistic_batch_scenario(self, xai_evaluator):
        """
        Test a realistic batch scenario with mixed results.
        Simulates what would happen in actual pipeline execution.
        """

        # Create realistic test data
        results = [
            # Batch 1: Good detections with bboxes
            self.create_realistic_result(
                "cat_001", has_bbox=True, bbox_coords=[60, 60, 120, 120]
            ),
            self.create_realistic_result(
                "dog_002", has_bbox=True, bbox_coords=[90, 80, 180, 160]
            ),
            # Batch 1: Image without bbox (e.g., classification-only dataset)
            self.create_realistic_result("landscape_003", has_bbox=False),
            # Batch 1: More images with bboxes
            self.create_realistic_result(
                "bird_004", has_bbox=True, bbox_coords=[70, 50, 150, 130]
            ),
            self.create_realistic_result(
                "car_005", has_bbox=True, bbox_coords=[40, 100, 180, 200]
            ),
            # Batch 1: Another without bbox
            self.create_realistic_result("abstract_006", has_bbox=False),
            # Batch 1: Final image with bbox
            self.create_realistic_result(
                "person_007", has_bbox=True, bbox_coords=[85, 70, 140, 180]
            ),
        ]

        # Expected pattern: [has_metrics, has_metrics, None, has_metrics, has_metrics, None, has_metrics]
        expected_pattern = [True, True, False, True, True, False, True]

        # Run batch evaluation
        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # Verify results
        assert len(batch_metrics) == len(results)

        metrics_collected = []
        for i, (result, metrics, should_have_metrics) in enumerate(
            zip(results, batch_metrics, expected_pattern)
        ):
            print(
                f"Image {i}: {result.image_name} - has_bbox: {result.has_bbox} - metrics: {metrics is not None}"
            )

            if should_have_metrics:
                assert (
                    metrics is not None
                ), f"Image {i} ({result.image_name}) should have metrics"
                assert isinstance(metrics, MetricResults)

                # Verify all expected metrics are present
                expected_metrics = ["iou", "point_game", "pixel_precision_recall"]
                for metric_name in expected_metrics:
                    assert (
                        metric_name in metrics.values
                    ), f"Missing {metric_name} for image {i}"

                metrics_collected.append(metrics)
            else:
                assert (
                    metrics is None
                ), f"Image {i} ({result.image_name}) should not have metrics"

        # Should have collected metrics for 5 images (indices 0, 1, 3, 4, 6)
        assert len(metrics_collected) == 5

        return results, batch_metrics

    def test_orchestrator_integration_simulation(self, xai_evaluator):
        """
        Simulate the full orchestrator workflow to test integration.
        """

        # Create test results similar to what orchestrator would generate
        results = [
            self.create_realistic_result(
                f"batch_image_{i:03d}",
                has_bbox=(i % 3 != 0),  # Every 3rd image has no bbox
                prediction_correct=(i % 4 != 0),
            )  # Every 4th prediction is wrong
            for i in range(10)
        ]

        # Simulate orchestrator's evaluate_results method
        correct_predictions = sum(1 for r in results if r.prediction_correct)
        total_processing_time = sum(r.processing_time for r in results)

        # Run batch metrics evaluation
        individual_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # Create summary from individual metrics (like orchestrator does)
        summary = xai_evaluator.create_summary_from_individual_metrics(
            results=results,
            individual_metrics=individual_metrics,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

        # Verify summary
        assert isinstance(summary, EvaluationSummary)
        assert summary.total_samples == len(results)
        assert summary.prediction_accuracy == correct_predictions / len(results)
        assert (
            abs(
                summary.average_processing_time - (total_processing_time / len(results))
            )
            < 1e-6
        )

        # Verify that metrics were properly aggregated
        samples_with_bbox = sum(1 for r in results if r.has_bbox)
        assert summary.samples_with_bbox == samples_with_bbox

        # Verify individual metrics alignment
        non_none_metrics = [m for m in individual_metrics if m is not None]
        assert len(non_none_metrics) == samples_with_bbox

        print(f"Processed {len(results)} images:")
        print(f"  - {samples_with_bbox} with bounding boxes (got metrics)")
        print(
            f"  - {len(results) - samples_with_bbox} without bounding boxes (no metrics)"
        )
        print(f"  - Accuracy: {summary.prediction_accuracy:.3f}")
        print(f"  - Avg processing time: {summary.average_processing_time:.3f}s")

        return summary, individual_metrics

    def test_metric_value_sanity_checks(self, xai_evaluator):
        """
        Test that the calculated metric values are reasonable.
        """

        # Create results with known bbox/attribution alignment
        results = []

        # Perfect alignment case
        perfect_result = self.create_realistic_result(
            "perfect", has_bbox=True, bbox_coords=[80, 80, 144, 144]
        )
        # Make attribution perfectly aligned with bbox
        perfect_result.attribution[:, 80:144, 80:144] = 1.0
        perfect_result.attribution[:, :80, :] = 0.0
        perfect_result.attribution[:, 144:, :] = 0.0
        perfect_result.attribution[:, :, :80] = 0.0
        perfect_result.attribution[:, :, 144:] = 0.0
        results.append(perfect_result)

        # No alignment case
        no_align_result = self.create_realistic_result(
            "no_align", has_bbox=True, bbox_coords=[80, 80, 144, 144]
        )
        # Make attribution completely misaligned
        no_align_result.attribution[:, 80:144, 80:144] = 0.0
        no_align_result.attribution[:, :80, :80] = 1.0  # Put signal in opposite corner
        results.append(no_align_result)

        # No bbox case
        results.append(self.create_realistic_result("no_bbox", has_bbox=False))

        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # Check perfect alignment metrics
        perfect_metrics = batch_metrics[0]
        assert perfect_metrics is not None
        perfect_iou = perfect_metrics.values["iou"]
        perfect_point_game = perfect_metrics.values["point_game"]

        print(
            f"Perfect alignment - IOU: {perfect_iou:.3f}, Point Game: {perfect_point_game}"
        )

        # Check no alignment metrics
        no_align_metrics = batch_metrics[1]
        assert no_align_metrics is not None
        no_align_iou = no_align_metrics.values["iou"]
        no_align_point_game = no_align_metrics.values["point_game"]

        print(
            f"No alignment - IOU: {no_align_iou:.3f}, Point Game: {no_align_point_game}"
        )

        # Perfect alignment should have better metrics than no alignment
        assert (
            perfect_iou >= no_align_iou
        ), "Perfect alignment should have better or equal IOU"

        # No bbox case should have no metrics
        assert batch_metrics[2] is None

    def test_error_resilience(self, xai_evaluator):
        """
        Test that the batch evaluation is resilient to various error conditions.
        """

        # Create results with potential problem cases
        results = []

        # Normal case
        results.append(self.create_realistic_result("normal", has_bbox=True))

        # Edge case: very small bbox
        small_bbox_result = self.create_realistic_result(
            "small_bbox", has_bbox=True, bbox_coords=[100, 100, 101, 101]
        )
        results.append(small_bbox_result)

        # Edge case: bbox at image boundary
        boundary_result = self.create_realistic_result(
            "boundary", has_bbox=True, bbox_coords=[0, 0, 50, 50]
        )
        results.append(boundary_result)

        # No bbox case
        results.append(self.create_realistic_result("no_bbox", has_bbox=False))

        # Another normal case
        results.append(self.create_realistic_result("normal2", has_bbox=True))

        # Should not crash and should return reasonable results
        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        assert len(batch_metrics) == len(results)

        # Check that we got reasonable results
        expected_non_none = [0, 1, 2, 4]  # Indices that should have metrics
        for i in expected_non_none:
            assert batch_metrics[i] is not None, f"Index {i} should have metrics"

        assert batch_metrics[3] is None, "Index 3 (no bbox) should have None metrics"


if __name__ == "__main__":
    # Run the tests
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
