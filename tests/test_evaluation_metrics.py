import unittest

import torch

from src.pipeline.pipeline_moduls.evaluation.base.metric_registry import MetricRegistry
from src.pipeline.pipeline_moduls.evaluation.metrics.iou_metric import IoUMetric
from src.pipeline.pipeline_moduls.evaluation.metrics.pixel_precision_recall import (
    PixelPrecisionRecall,
)
from src.pipeline.pipeline_moduls.evaluation.metrics.point_game_metric import (
    PointGameMetric,
)


class TestIoUMetric(unittest.TestCase):

    def setUp(self):
        self.threshold = 0.5
        self.metric = IoUMetric(threshold=self.threshold)

    def test_initialization_with_threshold(self):
        metric = IoUMetric(threshold=0.3)
        self.assertEqual(metric.threshold, 0.3)
        self.assertEqual(metric.get_name(), "IoU")

    def test_initialization_missing_threshold(self):
        with self.assertRaises(ValueError):
            IoUMetric()

    def test_initialization_invalid_threshold_type(self):
        with self.assertRaises(TypeError):
            IoUMetric(threshold="0.5")

    def test_perfect_overlap(self):
        heatmap = torch.ones(10, 10)
        ground_truth = torch.ones(10, 10)

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 1.0)

    def test_no_overlap(self):
        heatmap = torch.zeros(10, 10)
        ground_truth = torch.ones(10, 10)

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 0.0)

    def test_partial_overlap(self):
        heatmap = torch.zeros(10, 10)
        heatmap[:5, :5] = 1.0  # Top-left quadrant

        ground_truth = torch.zeros(10, 10)
        ground_truth[:5, :] = 1.0  # Top half

        score = self.metric.calculate(heatmap, ground_truth)

        # Intersection: 25 pixels, Union: 75 pixels
        expected_iou = 25 / 75
        self.assertAlmostEqual(score, expected_iou, places=5)

    def test_threshold_binarization(self):
        heatmap = torch.tensor([[0.3, 0.7], [0.4, 0.8]])
        ground_truth = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

        score = self.metric.calculate(heatmap, ground_truth)

        # With threshold 0.5: heatmap becomes [[0, 1], [0, 1]]
        # Intersection: 2 pixels, Union: 3 pixels
        expected_iou = 2 / 3
        self.assertAlmostEqual(score, expected_iou, places=5)

    def test_3d_input_handling(self):
        # Test with 3D heatmap (averaged across channels)
        heatmap = torch.ones(3, 10, 10)
        ground_truth = torch.ones(1, 10, 10)

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 1.0)

    def test_empty_masks(self):
        heatmap = torch.zeros(10, 10)
        ground_truth = torch.zeros(10, 10)

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 0.0)


class TestPointGameMetric(unittest.TestCase):

    def setUp(self):
        self.metric = PointGameMetric()

    def test_initialization(self):
        self.assertEqual(self.metric.get_name(), "point_game")

    def test_max_point_inside_mask(self):
        heatmap = torch.zeros(5, 5)
        heatmap[2, 3] = 1.0  # Max point at (3, 2)

        ground_truth = torch.zeros(5, 5)
        ground_truth[2, 3] = 1.0  # Mask covers the max point

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 1.0)

    def test_max_point_outside_mask(self):
        heatmap = torch.zeros(5, 5)
        heatmap[2, 3] = 1.0  # Max point at (3, 2)

        ground_truth = torch.zeros(5, 5)
        ground_truth[1, 1] = 1.0  # Mask doesn't cover the max point

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 0.0)

    def test_find_max_point(self):
        heatmap = torch.zeros(3, 4)
        heatmap[1, 2] = 0.8

        max_point = self.metric._find_max_point(heatmap)
        self.assertEqual(max_point, (2, 1))  # (x, y) format

    def test_find_max_point_multiple_maxima(self):
        heatmap = torch.zeros(3, 3)
        heatmap[0, 0] = 1.0
        heatmap[2, 2] = 1.0  # Two equal maxima

        max_point = self.metric._find_max_point(heatmap)
        # Should return the first occurrence
        self.assertEqual(max_point, (0, 0))

    def test_point_in_mask_true(self):
        mask = torch.zeros(3, 3)
        mask[1, 2] = 1.0

        result = self.metric._point_in_mask((2, 1), mask)
        self.assertTrue(result)

    def test_point_in_mask_false(self):
        mask = torch.zeros(3, 3)
        mask[1, 2] = 1.0

        result = self.metric._point_in_mask((0, 0), mask)
        self.assertFalse(result)

    def test_point_out_of_bounds(self):
        mask = torch.zeros(3, 3)

        result = self.metric._point_in_mask((5, 5), mask)
        self.assertFalse(result)

    def test_3d_input_handling(self):
        heatmap = torch.zeros(3, 4, 5)
        heatmap[:, 2, 3] = 1.0  # Max across all channels at (3, 2)

        ground_truth = torch.zeros(1, 4, 5)
        ground_truth[0, 2, 3] = 1.0

        score = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(score, 1.0)


class TestPixelPrecisionRecallMetric(unittest.TestCase):

    def setUp(self):
        self.threshold = 0.5
        self.metric = PixelPrecisionRecall(threshold=self.threshold)

    def test_initialization(self):
        self.assertEqual(self.metric.threshold, 0.5)
        self.assertEqual(self.metric.get_name(), "pixel_precision_recall")

    def test_perfect_precision_and_recall(self):
        heatmap = torch.ones(3, 3)
        ground_truth = torch.ones(3, 3)

        precision, recall = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_zero_precision_and_recall(self):
        heatmap = torch.zeros(3, 3)
        ground_truth = torch.ones(3, 3)

        precision, recall = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_partial_precision_and_recall(self):
        # Heatmap predicts 4 pixels as positive
        heatmap = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        # Ground truth has 6 positive pixels
        ground_truth = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

        precision, recall = self.metric.calculate(heatmap, ground_truth)

        # True positives: 4, False positives: 0, False negatives: 2
        # Precision = 4/4 = 1.0, Recall = 4/6 = 0.667
        self.assertEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 4 / 6, places=5)

    def test_no_ground_truth_positives(self):
        heatmap = torch.ones(3, 3)
        ground_truth = torch.zeros(3, 3)

        precision, recall = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)  # No positives to recall

    def test_no_predicted_positives(self):
        heatmap = torch.zeros(3, 3)
        ground_truth = torch.ones(3, 3)

        precision, recall = self.metric.calculate(heatmap, ground_truth)
        self.assertEqual(precision, 0.0)  # No predictions made
        self.assertEqual(recall, 0.0)


class TestMetricRegistry(unittest.TestCase):

    def test_iou_metric_registration(self):
        metric_class = MetricRegistry.get_metric("iou")
        self.assertEqual(metric_class, IoUMetric)

    def test_point_game_metric_registration(self):
        metric_class = MetricRegistry.get_metric("point_game")
        self.assertEqual(metric_class, PointGameMetric)

    def test_pixel_precision_recall_registration(self):
        metric_class = MetricRegistry.get_metric("pixel_precision_recall")
        self.assertEqual(metric_class, PixelPrecisionRecall)

    def test_invalid_metric_name(self):
        with self.assertRaises(KeyError):
            MetricRegistry.get_metric("nonexistent_metric")

    def test_list_available_metrics(self):
        available = MetricRegistry.list_available()
        self.assertIn("iou", available)
        self.assertIn("point_game", available)
        self.assertIn("pixel_precision_recall", available)


class TestMetricIntegration(unittest.TestCase):

    def test_all_metrics_on_same_data(self):
        # Create test data
        heatmap = torch.tensor([[0.8, 0.3, 0.1], [0.6, 0.9, 0.2], [0.1, 0.4, 0.7]])

        ground_truth = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Test IoU
        iou_metric = IoUMetric(threshold=0.5)
        iou_score = iou_metric.calculate(heatmap, ground_truth)
        self.assertIsInstance(iou_score, float)
        self.assertGreaterEqual(iou_score, 0.0)
        self.assertLessEqual(iou_score, 1.0)

        # Test Point Game
        pg_metric = PointGameMetric()
        pg_score = pg_metric.calculate(heatmap, ground_truth)
        self.assertIn(pg_score, [0.0, 1.0])

        # Test Pixel Precision Recall
        ppr_metric = PixelPrecisionRecall(threshold=0.5)
        precision, recall = ppr_metric.calculate(heatmap, ground_truth)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)

    def test_batch_processing(self):
        # Test metrics with different batch sizes and shapes
        batch_size = 5
        height, width = 32, 32

        heatmaps = torch.rand(batch_size, height, width)
        ground_truths = torch.randint(0, 2, (batch_size, height, width)).float()

        iou_metric = IoUMetric(threshold=0.5)
        pg_metric = PointGameMetric()

        iou_scores = []
        pg_scores = []

        for i in range(batch_size):
            iou_score = iou_metric.calculate(heatmaps[i], ground_truths[i])
            pg_score = pg_metric.calculate(heatmaps[i], ground_truths[i])

            iou_scores.append(iou_score)
            pg_scores.append(pg_score)

        # Verify all scores are valid
        for score in iou_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        for score in pg_scores:
            self.assertIn(score, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
