from pathlib import Path

import pytest
import torch

from src.pipeline.control.utils.dataclasses.xai_explanation_result import (
    XAIExplanationResult,
)
from src.pipeline.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults
from src.pipeline.pipeline_moduls.evaluation.xai_evaluator import XAIEvaluator


class TestBatchMetricAssignment:
    """
    Test suite to verify that batch metric evaluation correctly assigns
    metrics to the right images, even when some images have no bounding boxes.
    """

    def create_test_result(
        self,
        image_name: str,
        has_bbox: bool = True,
        prediction_correct: bool = True,
        image_size: tuple = (224, 224),
    ) -> XAIExplanationResult:
        """Create a test XAI result with controllable bbox presence."""

        # Create dummy image and attribution
        image = torch.randn(3, *image_size)
        attribution = torch.randn(3, *image_size)

        # Create bbox or empty tensor based on has_bbox
        if has_bbox:
            # Create a valid bounding box (x1, y1, x2, y2 format)
            bbox = torch.tensor([50.0, 50.0, 150.0, 150.0])
        else:
            # Empty bbox tensor
            bbox = torch.tensor([])

        return XAIExplanationResult(
            image=image,
            image_path=Path(f"/fake/path/{image_name}.jpg"),
            image_name=image_name,
            predicted_class=1,
            prediction_confidence=0.95,
            true_label=1 if prediction_correct else 0,
            prediction_correct=prediction_correct,
            topk_predictions=[1, 2, 3],
            topk_confidences=[0.95, 0.03, 0.02],
            attribution=attribution,
            explainer_result=None,
            explainer_name="test_explainer",
            explainer_params={},
            has_bbox=has_bbox,
            bbox=bbox,
            model_name="test_model",
            processing_time=0.1,
            timestamp="1234567890",
        )

    @pytest.fixture
    def xai_evaluator(self):
        """Create XAI evaluator with basic metrics."""
        return XAIEvaluator(
            metric_names=["iou", "point_game"],
            metric_kwargs={"iou": {}, "point_game": {}},
        )

    def test_batch_assignment_with_mixed_bbox_presence(self, xai_evaluator):
        """
        Test that metrics are correctly assigned when some images lack bounding boxes.

        Test scenario:
        - Image 0: HAS bbox -> should get metrics
        - Image 1: NO bbox -> should get None
        - Image 2: HAS bbox -> should get metrics
        - Image 3: NO bbox -> should get None
        - Image 4: HAS bbox -> should get metrics
        """

        # Create test results with mixed bbox presence
        results = [
            self.create_test_result("image_0", has_bbox=True),  # Index 0: HAS bbox
            self.create_test_result("image_1", has_bbox=False),  # Index 1: NO bbox
            self.create_test_result("image_2", has_bbox=True),  # Index 2: HAS bbox
            self.create_test_result("image_3", has_bbox=False),  # Index 3: NO bbox
            self.create_test_result("image_4", has_bbox=True),  # Index 4: HAS bbox
        ]

        # Run batch evaluation
        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # Verify correct length
        assert len(batch_metrics) == len(
            results
        ), f"Expected {len(results)} results, got {len(batch_metrics)}"

        # Verify correct assignment pattern
        expected_pattern = [
            True,
            False,
            True,
            False,
            True,
        ]  # True = has metrics, False = None

        for i, (result, metrics, expected_has_metrics) in enumerate(
            zip(results, batch_metrics, expected_pattern)
        ):
            if expected_has_metrics:
                assert (
                    metrics is not None
                ), f"Image {i} ({result.image_name}) should have metrics but got None"
                assert isinstance(
                    metrics, MetricResults
                ), f"Image {i} should have MetricResults, got {type(metrics)}"
                assert (
                    metrics.values is not None
                ), f"Image {i} should have metric values"

                # Verify metrics contain expected keys
                expected_metrics = ["iou", "point_game"]
                for metric_name in expected_metrics:
                    assert (
                        metric_name in metrics.values
                    ), f"Image {i} missing metric '{metric_name}'"

            else:
                assert (
                    metrics is None
                ), f"Image {i} ({result.image_name}) should have None but got {metrics}"

    def test_batch_assignment_all_with_bbox(self, xai_evaluator):
        """Test batch assignment when all images have bounding boxes."""

        results = [
            self.create_test_result(f"image_{i}", has_bbox=True) for i in range(5)
        ]

        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # All should have metrics
        assert len(batch_metrics) == len(results)
        for i, metrics in enumerate(batch_metrics):
            assert metrics is not None, f"Image {i} should have metrics"
            assert isinstance(metrics, MetricResults)

    def test_batch_assignment_none_with_bbox(self, xai_evaluator):
        """Test batch assignment when no images have bounding boxes."""

        results = [
            self.create_test_result(f"image_{i}", has_bbox=False) for i in range(5)
        ]

        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # None should have metrics
        assert len(batch_metrics) == len(results)
        for i, metrics in enumerate(batch_metrics):
            assert metrics is None, f"Image {i} should have None but got {metrics}"

    def test_batch_assignment_complex_pattern(self, xai_evaluator):
        """Test with a more complex pattern of bbox presence."""

        # Pattern: [True, False, False, True, True, False, True]
        bbox_pattern = [True, False, False, True, True, False, True]

        results = [
            self.create_test_result(f"image_{i}", has_bbox=has_bbox)
            for i, has_bbox in enumerate(bbox_pattern)
        ]

        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        assert len(batch_metrics) == len(results)

        for i, (has_bbox, metrics) in enumerate(zip(bbox_pattern, batch_metrics)):
            if has_bbox:
                assert metrics is not None, f"Image {i} with bbox should have metrics"
                assert isinstance(metrics, MetricResults)
            else:
                assert (
                    metrics is None
                ), f"Image {i} without bbox should have None metrics"

    def test_metric_values_consistency(self, xai_evaluator):
        """
        Test that the actual metric values make sense and are consistent.
        This verifies that not only the assignment is correct, but also
        that the metrics are actually calculated properly.
        """

        # Create results with known patterns
        results = [
            self.create_test_result("image_0", has_bbox=True),  # Should get metrics
            self.create_test_result("image_1", has_bbox=False),  # Should get None
            self.create_test_result("image_2", has_bbox=True),  # Should get metrics
        ]

        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        # Extract metrics for images with bbox
        metrics_0 = batch_metrics[0]
        metrics_2 = batch_metrics[2]

        assert metrics_0 is not None
        assert metrics_2 is not None

        # Verify metric values are in reasonable ranges
        for metrics in [metrics_0, metrics_2]:
            iou_value = metrics.values.get("iou")
            point_game_value = metrics.values.get("point_game")

            assert iou_value is not None, "IOU metric should be present"
            assert point_game_value is not None, "Point game metric should be present"

            # IOU should be between 0 and 1
            assert 0.0 <= iou_value <= 1.0, f"IOU should be in [0,1], got {iou_value}"

            # Point game should be 0 or 1 (binary metric)
            assert point_game_value in [
                0.0,
                1.0,
            ], f"Point game should be 0 or 1, got {point_game_value}"

    def test_empty_results_list(self, xai_evaluator):
        """Test behavior with empty results list."""

        results = []
        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        assert batch_metrics == []

    def test_single_result_with_bbox(self, xai_evaluator):
        """Test with single result that has bbox."""

        results = [self.create_test_result("single_image", has_bbox=True)]
        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        assert len(batch_metrics) == 1
        assert batch_metrics[0] is not None
        assert isinstance(batch_metrics[0], MetricResults)

    def test_single_result_without_bbox(self, xai_evaluator):
        """Test with single result that has no bbox."""

        results = [self.create_test_result("single_image", has_bbox=False)]
        batch_metrics = xai_evaluator.evaluate_batch_metrics(results)

        assert len(batch_metrics) == 1
        assert batch_metrics[0] is None


if __name__ == "__main__":
    # Run the tests
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
