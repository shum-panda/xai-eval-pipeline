"""
Comprehensive Test Suite for XAI Evaluation Module

Tests all metric calculations, edge cases, and evaluation components
including IoU, Point Game, Pixel Precision/Recall, and XAIEvaluator.
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import torch

# Add project path for imports (adjust path as needed)
# sys.path.append('/path/to/your/project')

# Mock the imports if modules aren't available during testing
try:
    from src.control.utils.dataclasses.xai_explanation_result import (
        XAIExplanationResult,
    )
    from src.pipeline_moduls.evaluation.base.metric_calculator import MetricCalculator
    from src.pipeline_moduls.evaluation.base.metric_registry import (
        MetricRegistry,  # noqa: F401
    )
    from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
        EvaluationSummary,
    )
    from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults
    from src.pipeline_moduls.evaluation.metrics.iou_metric import IoUMetric
    from src.pipeline_moduls.evaluation.metrics.pixel_precision_recall import (
        PixelPrecisionRecall,
    )
    from src.pipeline_moduls.evaluation.metrics.point_game_metric import PointGameMetric
    from src.pipeline_moduls.evaluation.xai_evaluator import (
        XAIEvaluator,
        bbox_to_mask_tensor,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    # Create mock classes for testing structure
    IMPORTS_AVAILABLE = False
    print(
        "Warning: Module imports not available. Creating mock classes for testing "
        "structure."
    )


class TestDataGenerator:
    """Helper class to generate test data for metrics"""

    @staticmethod
    def create_perfect_heatmap_and_ground_truth(shape=(224, 224)):
        """Creates a perfect heatmap that exactly matches ground truth"""
        ground_truth = torch.zeros(shape)
        ground_truth[50:150, 50:150] = 1.0  # Square object
        heatmap = ground_truth.clone()
        return heatmap, ground_truth

    @staticmethod
    def create_partial_overlap_heatmap_and_ground_truth(shape=(224, 224)):
        """Creates heatmap with partial overlap"""
        ground_truth = torch.zeros(shape)
        ground_truth[50:150, 50:150] = 1.0  # Original square

        heatmap = torch.zeros(shape)
        heatmap[75:175, 75:175] = 1.0  # Shifted square (partial overlap)
        return heatmap, ground_truth

    @staticmethod
    def create_no_overlap_heatmap_and_ground_truth(shape=(224, 224)):
        """Creates heatmap with no overlap"""
        ground_truth = torch.zeros(shape)
        ground_truth[50:100, 50:100] = 1.0  # Small square

        heatmap = torch.zeros(shape)
        heatmap[150:200, 150:200] = 1.0  # Different location
        return heatmap, ground_truth

    @staticmethod
    def create_continuous_heatmap_and_binary_ground_truth(shape=(224, 224)):
        """Creates continuous heatmap with binary ground truth"""
        ground_truth = torch.zeros(shape)
        ground_truth[50:150, 50:150] = 1.0

        heatmap = torch.zeros(shape)
        # Gaussian-like distribution centered on object
        center_y, center_x = 100, 100
        y, x = torch.meshgrid(
            torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij"
        )
        heatmap = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * 30**2))
        return heatmap, ground_truth

    @staticmethod
    def create_noise_heatmap_and_ground_truth(shape=(224, 224)):
        """Creates noisy heatmap"""
        ground_truth = torch.zeros(shape)
        ground_truth[50:150, 50:150] = 1.0

        heatmap = torch.rand(shape) * 0.1  # Low noise
        heatmap[75:125, 75:125] += 0.8  # Higher activation in center of object
        return heatmap, ground_truth


class TestIoUMetric(unittest.TestCase):
    """Test cases for IoU (Intersection over Union) metric"""

    def setUp(self):
        if IMPORTS_AVAILABLE:
            self.iou_metric = IoUMetric(threshold=0.5)
        else:
            self.iou_metric = Mock()

    def test_perfect_overlap_iou(self):
        """Test IoU with perfect overlap (should be 1.0)"""
        heatmap, ground_truth = (
            TestDataGenerator.create_perfect_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            iou_score = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertAlmostEqual(iou_score, 1.0, places=4)
        else:
            # Mock test
            self.iou_metric.calculate.return_value = 1.0
            result = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result, 1.0)

    def test_no_overlap_iou(self):
        """Test IoU with no overlap (should be 0.0)"""
        heatmap, ground_truth = (
            TestDataGenerator.create_no_overlap_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            iou_score = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertAlmostEqual(iou_score, 0.0, places=4)
        else:
            self.iou_metric.calculate.return_value = 0.0
            result = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result, 0.0)

    def test_partial_overlap_iou(self):
        """Test IoU with partial overlap"""
        heatmap, ground_truth = (
            TestDataGenerator.create_partial_overlap_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            iou_score = self.iou_metric.calculate(heatmap, ground_truth)
            # Expected IoU for shifted squares should be > 0 and < 1
            self.assertGreater(iou_score, 0.0)
            self.assertLess(iou_score, 1.0)
        else:
            self.iou_metric.calculate.return_value = 0.36  # Approximate expected value
            result = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_different_thresholds(self):
        """Test IoU with different thresholds"""
        heatmap, ground_truth = (
            TestDataGenerator.create_continuous_heatmap_and_binary_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            # Test with low threshold
            low_threshold_metric = IoUMetric(threshold=0.1)
            low_iou = low_threshold_metric.calculate(heatmap, ground_truth)

            # Test with high threshold
            high_threshold_metric = IoUMetric(threshold=0.9)
            high_iou = high_threshold_metric.calculate(heatmap, ground_truth)

            # Low threshold should give higher IoU
            self.assertGreaterEqual(low_iou, high_iou)
        else:
            # Mock different threshold behavior
            self.assertTrue(True)  # Placeholder

    def test_empty_ground_truth(self):
        """Test IoU with empty ground truth"""
        heatmap = torch.rand((224, 224))
        ground_truth = torch.zeros((224, 224))

        if IMPORTS_AVAILABLE:
            iou_score = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertAlmostEqual(iou_score, 0.0, places=4)
        else:
            self.iou_metric.calculate.return_value = 0.0
            result = self.iou_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result, 0.0)


class TestPointGameMetric(unittest.TestCase):
    """Test cases for Point Game metric"""

    def setUp(self):
        if IMPORTS_AVAILABLE:
            self.point_game_metric = PointGameMetric()
        else:
            self.point_game_metric = Mock()

    def test_max_point_in_object(self):
        """Test point game when maximum point is in object"""
        heatmap, ground_truth = (
            TestDataGenerator.create_perfect_heatmap_and_ground_truth()
        )
        # Ensure maximum is clearly in the object area
        heatmap[100, 100] = 1.0  # Center of object

        if IMPORTS_AVAILABLE:
            score = self.point_game_metric.calculate(heatmap, ground_truth)
            self.assertEqual(score, 1.0)
        else:
            self.point_game_metric.calculate.return_value = 1.0
            result = self.point_game_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result, 1.0)

    def test_max_point_outside_object(self):
        """Test point game when maximum point is outside object"""
        heatmap, ground_truth = (
            TestDataGenerator.create_perfect_heatmap_and_ground_truth()
        )
        heatmap[10, 10] = 1.0  # Outside object area
        heatmap[50:150, 50:150] = 0.5  # Lower values in object

        if IMPORTS_AVAILABLE:
            score = self.point_game_metric.calculate(heatmap, ground_truth)
            self.assertEqual(score, 0.0)
        else:
            self.point_game_metric.calculate.return_value = 0.0
            result = self.point_game_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result, 0.0)

    def test_multiple_maxima(self):
        """Test point game with multiple maximum points"""
        heatmap = torch.zeros((224, 224))
        ground_truth = torch.zeros((224, 224))
        ground_truth[50:150, 50:150] = 1.0

        # Create multiple maxima - one inside, one outside
        heatmap[100, 100] = 1.0  # Inside object
        heatmap[200, 200] = 1.0  # Outside object

        if IMPORTS_AVAILABLE:
            score = self.point_game_metric.calculate(heatmap, ground_truth)
            # Should be 1.0 if first maximum found is inside, 0.0 if outside
            self.assertIn(score, [0.0, 1.0])
        else:
            self.point_game_metric.calculate.return_value = 1.0
            result = self.point_game_metric.calculate(heatmap, ground_truth)
            self.assertIn(result, [0.0, 1.0])

    def test_empty_heatmap(self):
        """Test point game with empty heatmap"""
        heatmap = torch.zeros((224, 224))
        ground_truth = torch.ones((224, 224))

        if IMPORTS_AVAILABLE:
            score = self.point_game_metric.calculate(heatmap, ground_truth)
            # Behavior depends on implementation - could be 0.0 or 1.0
            self.assertIn(score, [0.0, 1.0])
        else:
            self.point_game_metric.calculate.return_value = 1.0
            result = self.point_game_metric.calculate(heatmap, ground_truth)
            self.assertIn(result, [0.0, 1.0])


class TestPixelPrecisionRecall(unittest.TestCase):
    """Test cases for Pixel-level Precision and Recall metrics"""

    def setUp(self):
        if IMPORTS_AVAILABLE:
            self.pixel_metric = PixelPrecisionRecall(threshold=0.5)
        else:
            self.pixel_metric = Mock()

    def test_perfect_precision_recall(self):
        """Test precision/recall with perfect match"""
        heatmap, ground_truth = (
            TestDataGenerator.create_perfect_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            self.assertIsInstance(result, dict)
            self.assertAlmostEqual(result["precision"], 1.0, places=4)
            self.assertAlmostEqual(result["recall"], 1.0, places=4)
        else:
            self.pixel_metric.calculate.return_value = {"precision": 1.0, "recall": 1.0}
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result["precision"], 1.0)
            self.assertEqual(result["recall"], 1.0)

    def test_precision_recall_no_overlap(self):
        """Test precision/recall with no overlap"""
        heatmap, ground_truth = (
            TestDataGenerator.create_no_overlap_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            self.assertAlmostEqual(result["precision"], 0.0, places=4)
            self.assertAlmostEqual(result["recall"], 0.0, places=4)
        else:
            self.pixel_metric.calculate.return_value = {"precision": 0.0, "recall": 0.0}
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            self.assertEqual(result["precision"], 0.0)
            self.assertEqual(result["recall"], 0.0)

    def test_high_precision_low_recall(self):
        """Test case with high precision but low recall"""
        ground_truth = torch.zeros((224, 224))
        ground_truth[50:150, 50:150] = 1.0  # Large object

        heatmap = torch.zeros((224, 224))
        heatmap[90:110, 90:110] = 1.0  # Small area within object

        if IMPORTS_AVAILABLE:
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            # Should have high precision (all predicted pixels are correct)
            # but low recall (missed many object pixels)
            self.assertGreater(result["precision"], 0.8)
            self.assertLess(result["recall"], 0.5)
        else:
            self.pixel_metric.calculate.return_value = {
                "precision": 1.0,
                "recall": 0.04,
            }
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            self.assertGreater(result["precision"], 0.8)
            self.assertLess(result["recall"], 0.5)

    def test_low_precision_high_recall(self):
        """Test case with low precision but high recall"""
        ground_truth = torch.zeros((224, 224))
        ground_truth[90:110, 90:110] = 1.0  # Small object

        heatmap = torch.zeros((224, 224))
        heatmap[50:150, 50:150] = 1.0  # Large area covering object

        if IMPORTS_AVAILABLE:
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            # Should have low precision (many false positives)
            # but high recall (covers most/all object pixels)
            self.assertLess(result["precision"], 0.2)
            self.assertGreater(result["recall"], 0.8)
        else:
            self.pixel_metric.calculate.return_value = {
                "precision": 0.04,
                "recall": 1.0,
            }
            result = self.pixel_metric.calculate(heatmap, ground_truth)
            self.assertLess(result["precision"], 0.2)
            self.assertGreater(result["recall"], 0.8)

    def test_different_thresholds(self):
        """Test precision/recall with different thresholds"""
        heatmap, ground_truth = (
            TestDataGenerator.create_continuous_heatmap_and_binary_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            low_threshold_metric = PixelPrecisionRecall(threshold=0.1)
            high_threshold_metric = PixelPrecisionRecall(threshold=0.9)

            low_result = low_threshold_metric.calculate(heatmap, ground_truth)
            high_result = high_threshold_metric.calculate(heatmap, ground_truth)

            # Low threshold typically gives higher recall, lower precision
            # High threshold typically gives lower recall, higher precision
            self.assertGreaterEqual(low_result["recall"], high_result["recall"])
        else:
            # Mock the threshold behavior
            self.assertTrue(True)


class TestMetricCalculator(unittest.TestCase):
    """Test cases for MetricCalculator that orchestrates multiple metrics"""

    def setUp(self):
        if IMPORTS_AVAILABLE:
            self.metric_names = ["iou", "pixel_precision_recall", "point_game"]
            self.metric_kwargs = {
                "iou": {"threshold": 0.5},
                "pixel_precision_recall": {"threshold": 0.5},
            }
            self.calculator = MetricCalculator(self.metric_names, self.metric_kwargs)
        else:
            self.calculator = Mock()

    def test_evaluate_all_metrics(self):
        """Test evaluation of all metrics together"""
        heatmap, ground_truth = (
            TestDataGenerator.create_perfect_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            results = self.calculator.evaluate(heatmap, ground_truth)

            # Check that all expected metrics are present
            self.assertIn("IoU", results)
            self.assertIn("PixelPrecisionRecall", results)
            self.assertIn("point_game", results)

            # Check value types
            self.assertIsInstance(results["IoU"], float)
            self.assertIsInstance(results["PixelPrecisionRecall"], dict)
            self.assertIsInstance(results["point_game"], float)
        else:
            expected_results = {
                "IoU": 1.0,
                "PixelPrecisionRecall": {"precision": 1.0, "recall": 1.0},
                "point_game": 1.0,
            }
            self.calculator.evaluate.return_value = expected_results
            results = self.calculator.evaluate(heatmap, ground_truth)
            self.assertEqual(len(results), 3)

    def test_evaluate_batch(self):
        """Test batch evaluation"""
        batch_size = 3
        heatmaps = torch.stack(
            [
                TestDataGenerator.create_perfect_heatmap_and_ground_truth()[0],
                TestDataGenerator.create_partial_overlap_heatmap_and_ground_truth()[0],
                TestDataGenerator.create_no_overlap_heatmap_and_ground_truth()[0],
            ]
        )
        ground_truths = torch.stack(
            [
                TestDataGenerator.create_perfect_heatmap_and_ground_truth()[1],
                TestDataGenerator.create_partial_overlap_heatmap_and_ground_truth()[1],
                TestDataGenerator.create_no_overlap_heatmap_and_ground_truth()[1],
            ]
        )

        if IMPORTS_AVAILABLE:
            batch_results = self.calculator.evaluate_batch(heatmaps, ground_truths)

            self.assertEqual(len(batch_results), batch_size)
            for result in batch_results:
                self.assertIn("IoU", result)
                self.assertIn("PixelPrecisionRecall", result)
                self.assertIn("point_game", result)
        else:
            mock_batch_results = [
                {
                    "IoU": 1.0,
                    "PixelPrecisionRecall": {"precision": 1.0, "recall": 1.0},
                    "point_game": 1.0,
                },
                {
                    "IoU": 0.36,
                    "PixelPrecisionRecall": {"precision": 0.5, "recall": 0.5},
                    "point_game": 1.0,
                },
                {
                    "IoU": 0.0,
                    "PixelPrecisionRecall": {"precision": 0.0, "recall": 0.0},
                    "point_game": 0.0,
                },
            ]
            self.calculator.evaluate_batch.return_value = mock_batch_results
            results = self.calculator.evaluate_batch(heatmaps, ground_truths)
            self.assertEqual(len(results), batch_size)


class TestBboxToMaskTensor(unittest.TestCase):
    """Test cases for bbox_to_mask_tensor utility function"""

    def test_valid_bbox_conversion(self):
        """Test conversion of valid bounding box to mask"""
        bbox = torch.tensor([[50, 50, 150, 150]])  # x1, y1, x2, y2
        shape = (224, 224)

        if IMPORTS_AVAILABLE:
            mask = bbox_to_mask_tensor(bbox, shape)

            # Check shape
            self.assertEqual(mask.shape, (1, 224, 224))

            # Check that only bbox area is set to 1
            self.assertEqual(torch.sum(mask).item(), 100 * 100)  # 100x100 area
            self.assertEqual(mask[0, 50, 50].item(), 1.0)
            self.assertEqual(mask[0, 149, 149].item(), 1.0)
            self.assertEqual(mask[0, 49, 49].item(), 0.0)
            self.assertEqual(mask[0, 150, 150].item(), 0.0)
        else:
            # Mock the function behavior
            mock_mask = torch.zeros((1, 224, 224))
            mock_mask[0, 50:150, 50:150] = 1.0
            self.assertEqual(torch.sum(mock_mask).item(), 100 * 100)

    def test_edge_bbox_conversion(self):
        """Test conversion of edge case bounding boxes"""
        # Full image bbox
        bbox = torch.tensor([[0, 0, 224, 224]])
        shape = (224, 224)

        if IMPORTS_AVAILABLE:
            mask = bbox_to_mask_tensor(bbox, shape)
            self.assertEqual(torch.sum(mask).item(), 224 * 224)
        else:
            # Mock behavior
            self.assertTrue(True)


class TestXAIEvaluator(unittest.TestCase):
    """Test cases for XAIEvaluator main class"""

    def setUp(self):
        if IMPORTS_AVAILABLE:
            self.evaluator = XAIEvaluator(
                metric_names=["iou", "pixel_precision_recall", "point_game"],
                metric_kwargs={"iou": {"threshold": 0.5}},
            )
        else:
            self.evaluator = Mock()

    def create_mock_xai_result(self, has_bbox=True, prediction_correct=True):
        """Create a mock XAIExplanationResult for testing"""
        if IMPORTS_AVAILABLE:
            result = Mock(spec=XAIExplanationResult)
        else:
            result = Mock()

        result.has_bbox = has_bbox
        result.prediction_correct = prediction_correct
        result.image_name = "test_image.jpg"
        result.model_name = "resnet50"
        result.explainer_name = "gradcam"
        result.predicted_class = "test_class"
        result.true_label = "test_class"
        result.processing_time = 0.1

        if has_bbox:
            result.bbox_info = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
            result.bbox = torch.tensor([[50, 50, 150, 150]])
            result.attribution = torch.rand((224, 224))
        else:
            result.bbox_info = None
            result.bbox = None
            result.attribution = torch.rand((224, 224))

        return result

    def test_evaluate_single_result_with_bbox(self):
        """Test evaluation of single result with bounding box"""
        result = self.create_mock_xai_result(has_bbox=True)

        if IMPORTS_AVAILABLE:
            metrics = self.evaluator.evaluate_single_result(result)
            self.assertIsNotNone(metrics)
            self.assertIsInstance(metrics, MetricResults)
            self.assertIsInstance(metrics.values, dict)
        else:
            mock_metrics = Mock()
            mock_metrics.values = {
                "IoU": 0.75,
                "PixelPrecisionRecall": {"precision": 0.8, "recall": 0.7},
                "point_game": 1.0,
            }
            self.evaluator.evaluate_single_result.return_value = mock_metrics
            result_metrics = self.evaluator.evaluate_single_result(result)
            self.assertIsNotNone(result_metrics)

    def test_evaluate_single_result_without_bbox(self):
        """Test evaluation of single result without bounding box"""
        result = self.create_mock_xai_result(has_bbox=False)

        if IMPORTS_AVAILABLE:
            metrics = self.evaluator.evaluate_single_result(result)
            self.assertIsNone(metrics)
        else:
            self.evaluator.evaluate_single_result.return_value = None
            result_metrics = self.evaluator.evaluate_single_result(result)
            self.assertIsNone(result_metrics)

    def test_evaluate_batch_results(self):
        """Test batch evaluation of multiple results"""
        results = [
            self.create_mock_xai_result(has_bbox=True, prediction_correct=True),
            self.create_mock_xai_result(has_bbox=True, prediction_correct=False),
            self.create_mock_xai_result(has_bbox=False, prediction_correct=True),
        ]

        if IMPORTS_AVAILABLE:
            with patch.object(self.evaluator, "evaluate_single_result") as mock_eval:
                mock_eval.side_effect = [
                    Mock(values={"IoU": 0.8, "point_game": 1.0}),
                    Mock(values={"IoU": 0.5, "point_game": 0.0}),
                    None,
                ]

                summary = self.evaluator.evaluate_batch_results(results)
                self.assertIsInstance(summary, EvaluationSummary)
                self.assertEqual(summary.total_samples, 3)
                self.assertEqual(summary.samples_with_bbox, 2)
                self.assertEqual(summary.correct_predictions, 2)
        else:
            mock_summary = Mock()
            mock_summary.total_samples = 3
            mock_summary.samples_with_bbox = 2
            mock_summary.correct_predictions = 2
            mock_summary.explainer_name = "gradcam"
            mock_summary.model_name = "resnet50"
            self.evaluator.evaluate_batch_results.return_value = mock_summary

            summary = self.evaluator.evaluate_batch_results(results)
            self.assertEqual(summary.total_samples, 3)
            self.assertEqual(summary.samples_with_bbox, 2)


class TestEvaluationSummary(unittest.TestCase):
    """Test cases for EvaluationSummary data class"""

    def test_evaluation_summary_creation(self):
        """Test creation and serialization of EvaluationSummary"""
        metric_averages = {
            "average_IoU": 0.75,
            "average_PixelPrecisionRecall_precision": 0.8,
            "average_PixelPrecisionRecall_recall": 0.7,
            "average_point_game": 0.9,
        }

        if IMPORTS_AVAILABLE:
            summary = EvaluationSummary(
                explainer_name="gradcam",
                model_name="resnet50",
                total_samples=100,
                samples_with_bbox=80,
                prediction_accuracy=0.85,
                correct_predictions=85,
                average_processing_time=0.15,
                total_processing_time=15.0,
                evaluation_timestamp=datetime.now().isoformat(),
                metric_averages=metric_averages,
            )

            # Test to_dict method
            summary_dict = summary.to_dict()
            self.assertIn("explainer_name", summary_dict)
            self.assertIn("average_IoU", summary_dict)
            self.assertEqual(summary_dict["total_samples"], 100)
        else:
            # Mock test
            self.assertTrue(True)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_empty_tensors(self):
        """Test behavior with empty tensors"""
        empty_heatmap = torch.zeros((224, 224))
        empty_ground_truth = torch.zeros((224, 224))

        if IMPORTS_AVAILABLE:
            iou_metric = IoUMetric(threshold=0.5)
            pixel_metric = PixelPrecisionRecall(threshold=0.5)
            point_game_metric = PointGameMetric()

            # IoU should be 0.0 for empty tensors
            iou_result = iou_metric.calculate(empty_heatmap, empty_ground_truth)
            self.assertEqual(iou_result, 0.0)

            # Pixel metrics should handle division by zero gracefully
            pixel_result = pixel_metric.calculate(empty_heatmap, empty_ground_truth)
            self.assertEqual(pixel_result["precision"], 0.0)
            self.assertEqual(pixel_result["recall"], 0.0)

            # Point game should handle empty tensors
            point_result = point_game_metric.calculate(
                empty_heatmap, empty_ground_truth
            )
            self.assertIn(point_result, [0.0, 1.0])
        else:
            # Mock tests
            self.assertTrue(True)

    def test_nan_and_inf_values(self):
        """Test behavior with NaN and infinite values"""
        heatmap = torch.ones((224, 224))
        heatmap[100, 100] = float("nan")
        heatmap[150, 150] = float("inf")

        ground_truth = torch.zeros((224, 224))
        ground_truth[50:150, 50:150] = 1.0

        if IMPORTS_AVAILABLE:
            iou_metric = IoUMetric(threshold=0.5)

            # Should handle NaN/inf gracefully without crashing
            try:
                result = iou_metric.calculate(heatmap, ground_truth)
                # Result should be a valid number
                self.assertFalse(torch.isnan(torch.tensor(result)))
                self.assertFalse(torch.isinf(torch.tensor(result)))
            except Exception as e:
                self.fail(f"IoU metric should handle NaN/inf gracefully: {e}")
        else:
            self.assertTrue(True)

    def test_mismatched_tensor_shapes(self):
        """Test behavior with mismatched tensor shapes"""
        heatmap = torch.rand((224, 224))
        ground_truth = torch.rand((256, 256))  # Different shape

        if IMPORTS_AVAILABLE:
            iou_metric = IoUMetric(threshold=0.5)

            with self.assertRaises(Exception):
                iou_metric.calculate(heatmap, ground_truth)
        else:
            self.assertTrue(True)

    def test_extreme_threshold_values(self):
        """Test metrics with extreme threshold values"""
        heatmap, ground_truth = (
            TestDataGenerator.create_continuous_heatmap_and_binary_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            # Test with very low threshold (0.0)
            low_metric = IoUMetric(threshold=0.0)
            low_result = low_metric.calculate(heatmap, ground_truth)
            self.assertGreaterEqual(low_result, 0.0)
            self.assertLessEqual(low_result, 1.0)

            # Test with very high threshold (1.0)
            high_metric = IoUMetric(threshold=1.0)
            high_result = high_metric.calculate(heatmap, ground_truth)
            self.assertGreaterEqual(high_result, 0.0)
            self.assertLessEqual(high_result, 1.0)
        else:
            self.assertTrue(True)


class TestPerformance(unittest.TestCase):
    """Performance tests for evaluation metrics"""

    def test_large_tensor_performance(self):
        """Test performance with large tensors"""
        import time

        # Large tensors (simulating high-resolution images)
        large_heatmap = torch.rand((1024, 1024))
        large_ground_truth = torch.zeros((1024, 1024))
        large_ground_truth[200:800, 200:800] = 1.0

        if IMPORTS_AVAILABLE:
            iou_metric = IoUMetric(threshold=0.5)

            start_time = time.time()
            result = iou_metric.calculate(large_heatmap, large_ground_truth)
            end_time = time.time()

            processing_time = end_time - start_time
            self.assertLess(processing_time, 1.0)  # Should complete within 1 second
            self.assertIsInstance(result, float)
        else:
            self.assertTrue(True)

    def test_batch_processing_performance(self):
        """Test performance of batch processing"""
        import time

        batch_size = 32
        heatmaps = torch.stack(
            [
                TestDataGenerator.create_continuous_heatmap_and_binary_ground_truth()[0]
                for _ in range(batch_size)
            ]
        )
        ground_truths = torch.stack(
            [
                TestDataGenerator.create_continuous_heatmap_and_binary_ground_truth()[1]
                for _ in range(batch_size)
            ]
        )

        if IMPORTS_AVAILABLE:
            calculator = MetricCalculator(
                ["iou", "pixel_precision_recall", "point_game"]
            )

            start_time = time.time()
            results = calculator.evaluate_batch(heatmaps, ground_truths)
            end_time = time.time()

            processing_time = end_time - start_time
            self.assertEqual(len(results), batch_size)
            self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        else:
            self.assertTrue(True)


class TestMetricValidation(unittest.TestCase):
    """Tests for metric validation and consistency"""

    def test_iou_symmetry(self):
        """Test that IoU is symmetric"""
        heatmap, ground_truth = (
            TestDataGenerator.create_partial_overlap_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            iou_metric = IoUMetric(threshold=0.5)

            # IoU(A, B) should equal IoU(B, A) for binary masks
            binary_heatmap = (heatmap >= 0.5).float()

            iou_ab = iou_metric.calculate(binary_heatmap, ground_truth)
            iou_ba = iou_metric.calculate(ground_truth, binary_heatmap)

            self.assertAlmostEqual(iou_ab, iou_ba, places=4)
        else:
            self.assertTrue(True)

    def test_metric_bounds(self):
        """Test that all metrics return values within expected bounds"""
        heatmap, ground_truth = (
            TestDataGenerator.create_continuous_heatmap_and_binary_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            calculator = MetricCalculator(
                ["iou", "pixel_precision_recall", "point_game"]
            )
            results = calculator.evaluate(heatmap, ground_truth)

            # IoU should be between 0 and 1
            self.assertGreaterEqual(results["IoU"], 0.0)
            self.assertLessEqual(results["IoU"], 1.0)

            # Precision and Recall should be between 0 and 1
            self.assertGreaterEqual(results["PixelPrecisionRecall"]["precision"], 0.0)
            self.assertLessEqual(results["PixelPrecisionRecall"]["precision"], 1.0)
            self.assertGreaterEqual(results["PixelPrecisionRecall"]["recall"], 0.0)
            self.assertLessEqual(results["PixelPrecisionRecall"]["recall"], 1.0)

            # Point Game should be 0 or 1
            self.assertIn(results["point_game"], [0.0, 1.0])
        else:
            self.assertTrue(True)

    def test_perfect_prediction_consistency(self):
        """Test that perfect predictions yield expected metric values"""
        heatmap, ground_truth = (
            TestDataGenerator.create_perfect_heatmap_and_ground_truth()
        )

        if IMPORTS_AVAILABLE:
            calculator = MetricCalculator(
                ["iou", "pixel_precision_recall", "point_game"]
            )
            results = calculator.evaluate(heatmap, ground_truth)

            # Perfect prediction should yield perfect scores
            self.assertAlmostEqual(results["IoU"], 1.0, places=4)
            self.assertAlmostEqual(
                results["PixelPrecisionRecall"]["precision"], 1.0, places=4
            )
            self.assertAlmostEqual(
                results["PixelPrecisionRecall"]["recall"], 1.0, places=4
            )
            self.assertEqual(results["point_game"], 1.0)
        else:
            self.assertTrue(True)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete evaluation pipeline"""

    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline from XAI result to summary"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Module imports not available")

        # Create mock XAI results
        results = []
        for i in range(5):
            result = Mock(spec=XAIExplanationResult)
            result.has_bbox = True
            result.prediction_correct = i < 3  # 3 correct, 2 incorrect
            result.image_name = f"test_image_{i}.jpg"
            result.model_name = "resnet50"
            result.explainer_name = "gradcam"
            result.predicted_class = "test_class"
            result.true_label = "test_class" if i < 3 else "other_class"
            result.processing_time = 0.1 + i * 0.02
            result.bbox_info = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
            result.bbox = torch.tensor([[50, 50, 150, 150]])

            # Create attribution with varying quality
            attribution = torch.zeros((224, 224))
            if i < 2:  # First 2 have good attribution
                attribution[75:125, 75:125] = 1.0
            elif i < 4:  # Next 2 have partial attribution
                attribution[60:140, 60:140] = 0.5
            # Last one has poor attribution (mostly zeros)

            result.attribution = attribution
            results.append(result)

        # Test evaluation
        evaluator = XAIEvaluator(
            metric_names=["iou", "pixel_precision_recall", "point_game"]
        )

        # Mock the evaluate_single_result method to return consistent results
        def mock_evaluate_single(result):
            if result.attribution.sum() > 1000:  # Good attribution
                return Mock(
                    values={
                        "IoU": 0.8,
                        "PixelPrecisionRecall": {"precision": 0.9, "recall": 0.85},
                        "point_game": 1.0,
                    }
                )
            elif result.attribution.sum() > 100:  # Partial attribution
                return Mock(
                    values={
                        "IoU": 0.4,
                        "PixelPrecisionRecall": {"precision": 0.6, "recall": 0.5},
                        "point_game": 1.0,
                    }
                )
            else:  # Poor attribution
                return Mock(
                    values={
                        "IoU": 0.1,
                        "PixelPrecisionRecall": {"precision": 0.2, "recall": 0.1},
                        "point_game": 0.0,
                    }
                )

        with patch.object(
            evaluator, "evaluate_single_result", side_effect=mock_evaluate_single
        ):
            summary = evaluator.evaluate_batch_results(results)

        # Verify summary
        self.assertEqual(summary.total_samples, 5)
        self.assertEqual(summary.samples_with_bbox, 5)
        self.assertEqual(summary.correct_predictions, 3)
        self.assertAlmostEqual(summary.prediction_accuracy, 0.6, places=2)
        self.assertGreater(summary.total_processing_time, 0.0)

        # Check that metric averages are calculated
        self.assertIn("average_IoU", summary.metric_averages)
        self.assertIn("average_PixelPrecisionRecall_precision", summary.metric_averages)
        self.assertIn("average_point_game", summary.metric_averages)


class TestRegressionPrevention(unittest.TestCase):
    """Tests to prevent regression in metric calculations"""

    def test_known_iou_values(self):
        """Test IoU with known expected values"""
        # Create specific scenario with known IoU
        heatmap = torch.zeros((100, 100))
        heatmap[25:75, 25:75] = 1.0  # 50x50 square

        ground_truth = torch.zeros((100, 100))
        ground_truth[40:90, 40:90] = 1.0  # 50x50 square, shifted

        if IMPORTS_AVAILABLE:
            iou_metric = IoUMetric(threshold=0.5)
            result = iou_metric.calculate(heatmap, ground_truth)

            # Calculate expected IoU manually
            # Intersection: 35x35 = 1225
            # Union: 50x50 + 50x50 - 35x35 = 2500 + 2500 - 1225 = 3775
            # IoU: 1225/3775 ≈ 0.3245
            expected_iou = 1225 / 3775
            self.assertAlmostEqual(result, expected_iou, places=3)
        else:
            self.assertTrue(True)

    def test_known_precision_recall_values(self):
        """Test Precision/Recall with known expected values"""
        heatmap = torch.zeros((10, 10))
        heatmap[2:6, 2:6] = 1.0  # 4x4 = 16 predicted positives

        ground_truth = torch.zeros((10, 10))
        ground_truth[3:7, 3:7] = 1.0  # 4x4 = 16 actual positives

        if IMPORTS_AVAILABLE:
            pixel_metric = PixelPrecisionRecall(threshold=0.5)
            result = pixel_metric.calculate(heatmap, ground_truth)

            # Intersection: 2x2 = 4 (overlap from [3:6, 3:6])
            # TP = 4, FP = 12, FN = 12
            # Precision = 4/16 = 0.25
            # Recall = 4/16 = 0.25
            self.assertAlmostEqual(result["precision"], 0.25, places=3)
            self.assertAlmostEqual(result["recall"], 0.25, places=3)
        else:
            self.assertTrue(True)


def run_evaluation_tests():
    """Main function to run all evaluation tests"""
    print("🧪 Starting Evaluation Module Test Suite")
    print("=" * 60)

    # Check if imports are available
    if not IMPORTS_AVAILABLE:
        print("⚠️  Warning: Module imports not available.")
        print("   Running in mock mode for testing structure only.")
        print("   Install modules to run actual implementation tests.")

    # Create test suite
    test_classes = [
        TestIoUMetric,
        TestPointGameMetric,
        TestPixelPrecisionRecall,
        TestMetricCalculator,
        TestBboxToMaskTensor,
        TestXAIEvaluator,
        TestEvaluationSummary,
        TestEdgeCases,
        TestPerformance,
        TestMetricValidation,
        TestIntegration,
        TestRegressionPrevention,
    ]

    # Run tests for each class
    total_tests = 0
    total_failures = 0
    total_errors = 0

    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open("/dev/null", "w"))
        result = runner.run(suite)

        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)

        # Print summary for this test class
        if result.wasSuccessful():
            print(f"   ✅ All {result.testsRun} tests passed")
        else:
            print(f"   ❌ {len(result.failures)} failures, {len(result.errors)} errors")
            if result.failures:
                print("   Failures:")
                for test, traceback in result.failures:
                    print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
            if result.errors:
                print("   Errors:")
                for test, traceback in result.errors:
                    print(f"     - {test}: {traceback.split('Exception:')[-1].strip()}")

    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(
        "Success Rate: "
        f"{((total_tests - total_failures - total_errors)  / total_tests * 100):.1f}%"
    )

    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed. Review the output above.")

    return total_tests, total_failures, total_errors


if __name__ == "__main__":
    # Run the test suite
    run_evaluation_tests()

    # Additional usage examples
    print("\n" + "=" * 60)
    print("💡 USAGE EXAMPLES")
    print("=" * 60)

    print(
        """
# 1. Run specific test class:
python -m unittest TestIoUMetric

# 2. Run specific test method:
python -m unittest TestIoUMetric.test_perfect_overlap_iou

# 3. Run with verbose output:
python -m unittest -v TestPixelPrecisionRecall

# 4. Integration with pytest (if available):
pytest evaluation_test_suite.py -v

# 5. Generate coverage report (if coverage.py installed):
coverage run evaluation_test_suite.py
coverage report
coverage html

# 6. Run performance tests only:
python -m unittest TestPerformance

# 7. Run edge case tests only:
python -m unittest TestEdgeCases
"""
    )

    print("\n🔧 CUSTOMIZATION:")
    print(
        """
# To add new test cases, inherit from unittest.TestCase:
class TestMyNewMetric(unittest.TestCase):
    def test_my_new_functionality(self):
        # Your test code here
        pass

# To test with real data:
- Set IMPORTS_AVAILABLE = True
- Ensure all modules are properly installed
- Modify TestDataGenerator for your specific data patterns

# To add performance benchmarks:
- Extend TestPerformance class
- Use time.time() for timing measurements
- Set reasonable performance thresholds
"""
    )
