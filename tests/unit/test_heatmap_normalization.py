import unittest

import torch

from src.pipe.moduls.evaluation.base.heatmap_normalization import normalize_heatmap
from src.pipe.moduls.evaluation.metrics.iou_metric import IoUMetric
from src.pipe.moduls.evaluation.metrics.pixel_precision_recall import (
    PixelPrecisionRecall,
)


class TestNormalizeHeatmap(unittest.TestCase):

    def test_none_keeps_raw_values(self) -> None:
        heatmap = torch.tensor([[0.001, 0.004]])
        torch.testing.assert_close(normalize_heatmap(heatmap, "none"), heatmap)

    def test_minmax_scales_to_unit_range(self) -> None:
        heatmap = torch.tensor([[2.0, 4.0], [6.0, 10.0]])
        expected = torch.tensor([[0.0, 0.25], [0.5, 1.0]])
        torch.testing.assert_close(normalize_heatmap(heatmap, "minmax"), expected)

    def test_max_divides_by_absolute_maximum(self) -> None:
        heatmap = torch.tensor([[1.0, -4.0]])
        expected = torch.tensor([[0.25, -1.0]])
        torch.testing.assert_close(normalize_heatmap(heatmap, "max"), expected)

    def test_constant_heatmap_highlights_nothing_with_minmax(self) -> None:
        heatmap = torch.full((3, 3), 5.0)
        torch.testing.assert_close(
            normalize_heatmap(heatmap, "minmax"), torch.zeros(3, 3)
        )


class TestMetricsWithNormalization(unittest.TestCase):
    """The same map on a different scale must give the same score."""

    def setUp(self) -> None:
        self.heatmap = torch.zeros(10, 10)
        self.heatmap[:5, :5] = 1.0
        self.ground_truth = torch.zeros(10, 10)
        self.ground_truth[:5, :] = 1.0

    def test_iou_is_scale_invariant_with_minmax(self) -> None:
        metric = IoUMetric(threshold=0.3, normalization="minmax")
        small = metric.calculate(self.heatmap * 0.001, self.ground_truth)
        large = metric.calculate(self.heatmap * 1000.0, self.ground_truth)
        self.assertAlmostEqual(small, 25 / 50, places=5)
        self.assertAlmostEqual(large, 25 / 50, places=5)

    def test_iou_without_normalization_depends_on_scale(self) -> None:
        metric = IoUMetric(threshold=0.3)
        self.assertEqual(metric.calculate(self.heatmap * 0.001, self.ground_truth), 0.0)

    def test_precision_recall_is_scale_invariant_with_minmax(self) -> None:
        metric = PixelPrecisionRecall(threshold=0.3, normalization="minmax")
        result = metric.calculate(self.heatmap * 0.001, self.ground_truth)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_unknown_normalization_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            IoUMetric(threshold=0.3, normalization="zscore")


if __name__ == "__main__":
    unittest.main()
