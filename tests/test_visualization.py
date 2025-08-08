import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.pipe.control.utils.dataclasses.xai_explanation_result import (
    XAIExplanationResult,
)
from src.pipe.moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipe.moduls.evaluation.dataclass.metricresults import (
    MetricResults,
)
from src.pipe.moduls.visualization.visualisation import Visualiser


class TestVisualiser(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir) / "visualizations"
        self.save_path.mkdir()

        # Create a test image
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        test_image = Image.new("RGB", (224, 224), color="red")
        test_image.save(self.test_image_path)

        # Create mock XAI explanation result
        self.mock_result = XAIExplanationResult(
            image_path=str(self.test_image_path),
            image_name="test_image.jpg",
            model_name="ResNet50",
            explainer_name="GradCAM",
            predicted_class=5,
            true_label=5,
            prediction_correct=True,
            attribution=torch.randn(224, 224),
            has_bbox=True,
            bbox=torch.tensor([[50, 50, 150, 150]]),
            predicted_class_name="dog",
            true_label_name="dog",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_visualiser_initialization_default(self):
        vis = Visualiser()
        self.assertTrue(vis.show)
        self.assertIsNone(vis.save_path)

    def test_visualiser_initialization_with_params(self):
        vis = Visualiser(show=False, save_path=self.save_path)
        self.assertFalse(vis.show)
        self.assertEqual(vis.save_path, self.save_path)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_create_visualization_without_metrics(self, mock_close, mock_show):
        vis = Visualiser(show=True, save_path=None)

        result = vis.create_visualization(self.mock_result)

        self.assertIsNone(result)  # No save path provided
        mock_show.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_create_visualization_with_save_path(
        self, mock_savefig, mock_close, mock_show
    ):
        vis = Visualiser(show=False, save_path=self.save_path)

        result = vis.create_visualization(self.mock_result)

        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".png"))
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        mock_show.assert_not_called()

    def test_create_visualization_with_metric_results(self):
        mock_metrics = MetricResults(
            sample_id="test_sample",
            values={
                "IoU": 0.75,
                "point_game": 1.0,
                "PixelPrecisionRecall": {"precision": 0.85, "recall": 0.90},
            },
        )

        vis = Visualiser(show=False, save_path=None)

        with patch("matplotlib.pyplot.close"):
            result = vis.create_visualization(self.mock_result, metrics=mock_metrics)

        self.assertIsNone(result)  # No save path provided

    def test_create_visualization_with_evaluation_summary(self):
        mock_summary = Mock(spec=EvaluationSummary)
        mock_summary.metric_averages = {
            "average_IoU": 0.65,
            "average_point_game": 0.8,
            "average_PixelPrecisionRecall_precision": 0.70,
            "average_PixelPrecisionRecall_recall": 0.75,
        }

        vis = Visualiser(show=False, save_path=None)

        with patch("matplotlib.pyplot.close"):
            result = vis.create_visualization(self.mock_result, metrics=mock_summary)

        self.assertIsNone(result)  # No save path provided

    def test_prepare_attribution_for_heatmap_2d(self):
        vis = Visualiser()
        attribution = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = vis._prepare_attribution_for_heatmap(attribution)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        # Should be normalized to [0, 1]
        self.assertAlmostEqual(result.max(), 1.0, places=5)
        self.assertAlmostEqual(result.min(), 0.0, places=5)

    def test_prepare_attribution_for_heatmap_3d(self):
        vis = Visualiser()
        attribution = torch.randn(3, 4, 4)  # 3 channels

        result = vis._prepare_attribution_for_heatmap(attribution)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))  # Should be averaged to 2D

    def test_extract_individual_metrics_from_metric_results(self):
        vis = Visualiser()

        mock_metrics = MetricResults(
            sample_id="test",
            values={
                "IoU": 0.8,
                "point_game": 1.0,
                "PixelPrecisionRecall": {"precision": 0.75, "recall": 0.85},
            },
        )

        iou, pg, precision, recall = vis._extract_individual_metrics(mock_metrics)

        self.assertEqual(iou, 0.8)
        self.assertEqual(pg, 1.0)
        self.assertEqual(precision, 0.75)
        self.assertEqual(recall, 0.85)

    def test_extract_individual_metrics_none(self):
        vis = Visualiser()

        iou, pg, precision, recall = vis._extract_individual_metrics(None)

        self.assertEqual(iou, 0.0)
        self.assertEqual(pg, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_extract_from_metric_results_dict_values(self):
        vis = Visualiser()

        mock_metrics = Mock()
        mock_metrics.values = {
            "IoU": 0.9,
            "point_game": 0.5,
            "PixelPrecisionRecall": {"precision": 0.88, "recall": 0.92},
        }

        iou, pg, precision, recall = vis._extract_from_metric_results(mock_metrics)

        self.assertEqual(iou, 0.9)
        self.assertEqual(pg, 0.5)
        self.assertEqual(precision, 0.88)
        self.assertEqual(recall, 0.92)

    def test_extract_from_metric_results_attributes(self):
        vis = Visualiser()

        mock_metrics = Mock()
        mock_metrics.values = None
        mock_metrics.iou_score = 0.7
        mock_metrics.point_game_score = 0.6
        mock_metrics.pixel_precision = 0.8
        mock_metrics.pixel_recall = 0.85

        iou, pg, precision, recall = vis._extract_from_metric_results(mock_metrics)

        self.assertEqual(iou, 0.7)
        self.assertEqual(pg, 0.6)
        self.assertEqual(precision, 0.8)
        self.assertEqual(recall, 0.85)

    @patch("matplotlib.pyplot.Axes.text")
    def test_create_metrics_display(self, mock_text):
        vis = Visualiser()

        fig, ax = plt.subplots()

        vis._create_metrics_display(
            ax=ax,
            result=self.mock_result,
            iou_score=0.75,
            point_game_score=1.0,
            pixel_precision=0.85,
            pixel_recall=0.90,
        )

        mock_text.assert_called_once()

        # Check that text contains expected information
        # call_args = mock_text.call_args[1]
        text_content = mock_text.call_args[0][2]  # Third positional argument

        self.assertIn("test_image.jpg", text_content)
        self.assertIn("ResNet50", text_content)
        self.assertIn("GradCAM", text_content)
        self.assertIn("0.750", text_content)  # IoU score
        self.assertIn("1.000", text_content)  # Point game score
        self.assertIn("[+] Yes", text_content)  # Correct prediction

        plt.close(fig)

    def test_save_and_show_plot_directory_save_path(self):
        vis = Visualiser(show=False, save_path=self.save_path)

        fig, ax = plt.subplots()

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            result = vis._save_and_show_plot(fig, self.mock_result)

        expected_filename = "test_image.jpg_ResNet50_GradCAM_vis.png"
        self.assertTrue(result.endswith(expected_filename))
        mock_savefig.assert_called_once()

    def test_save_and_show_plot_file_save_path(self):
        file_path = self.save_path / "custom_name.png"
        vis = Visualiser(show=False, save_path=file_path)

        fig, ax = plt.subplots()

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            result = vis._save_and_show_plot(fig, self.mock_result)

        self.assertEqual(result, str(file_path))
        mock_savefig.assert_called_once()

    def test_save_and_show_plot_no_save_path(self):
        vis = Visualiser(show=False, save_path=None)

        fig, ax = plt.subplots()

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            result = vis._save_and_show_plot(fig, self.mock_result)

        self.assertIsNone(result)
        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.show")
    def test_save_and_show_plot_with_show(self, mock_show):
        vis = Visualiser(show=True, save_path=None)

        fig, ax = plt.subplots()

        result = vis._save_and_show_plot(fig, self.mock_result)

        mock_show.assert_called_once()
        self.assertIsNone(result)


class TestVisualisationWithoutBbox(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create a test image
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        test_image = Image.new("RGB", (224, 224), color="blue")
        test_image.save(self.test_image_path)

        # Create mock result without bbox
        self.mock_result = XAIExplanationResult(
            image_path=str(self.test_image_path),
            image_name="test_image.jpg",
            model_name="VGG16",
            explainer_name="IntegratedGradients",
            predicted_class=3,
            true_label=7,
            prediction_correct=False,
            attribution=torch.randn(3, 224, 224),  # 3-channel attribution
            has_bbox=False,
            bbox=None,
            predicted_class_name="cat",
            true_label_name="horse",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.close")
    def test_create_visualization_without_bbox(self, mock_close):
        vis = Visualiser(show=False, save_path=None)

        result = vis.create_visualization(self.mock_result)

        self.assertIsNone(result)  # No save path provided
        mock_close.assert_called_once()

    def test_multichannel_attribution_handling(self):
        vis = Visualiser()

        # Test with 3-channel attribution
        attribution = torch.randn(3, 224, 224)
        result = vis._prepare_attribution_for_heatmap(attribution)

        self.assertEqual(result.shape, (224, 224))  # Should be averaged to 2D


class TestVisualisationErrorHandling(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create mock result with non-existent image
        self.mock_result = XAIExplanationResult(
            image_path="/nonexistent/path.jpg",
            image_name="nonexistent.jpg",
            model_name="TestModel",
            explainer_name="TestExplainer",
            predicted_class=1,
            true_label=1,
            prediction_correct=True,
            attribution=torch.randn(224, 224),
            has_bbox=False,
            bbox=None,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_visualization_with_missing_image(self):
        vis = Visualiser(show=False, save_path=None)

        with self.assertRaises(Exception):
            vis.create_visualization(self.mock_result)

    def test_create_visualization_with_invalid_attribution(self):
        vis = Visualiser(show=False, save_path=None)

        # Create a valid image path
        test_image_path = Path(self.temp_dir) / "test.jpg"
        test_image = Image.new("RGB", (224, 224), color="green")
        test_image.save(test_image_path)

        self.mock_result.image_path = str(test_image_path)
        self.mock_result.attribution = torch.tensor([])  # Invalid attribution

        with self.assertRaises(Exception):
            vis.create_visualization(self.mock_result)


class TestVisualisationIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir) / "visualizations"
        self.save_path.mkdir()

        # Create test image
        self.test_image_path = Path(self.temp_dir) / "integration_test.jpg"
        test_image = Image.new("RGB", (224, 224), color="purple")
        test_image.save(self.test_image_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.show")
    def test_full_visualization_pipeline(self, mock_show):
        # Create comprehensive test data
        result = XAIExplanationResult(
            image_path=str(self.test_image_path),
            image_name="integration_test.jpg",
            model_name="ResNet50",
            explainer_name="GradCAM",
            predicted_class=10,
            true_label=10,
            prediction_correct=True,
            attribution=torch.abs(torch.randn(224, 224)),  # Positive attribution
            has_bbox=True,
            bbox=torch.tensor([[25, 25, 200, 200]]),
            predicted_class_name="frog",
            true_label_name="frog",
        )

        metrics = MetricResults(
            sample_id="integration_test",
            values={
                "IoU": 0.82,
                "point_game": 1.0,
                "PixelPrecisionRecall": {"precision": 0.78, "recall": 0.85},
            },
        )

        # Test with saving
        vis_save = Visualiser(show=False, save_path=self.save_path)
        save_result = vis_save.create_visualization(result, metrics)

        self.assertIsNotNone(save_result)
        self.assertTrue(Path(save_result).exists())
        self.assertTrue(save_result.endswith(".png"))

        # Test with showing
        vis_show = Visualiser(show=True, save_path=None)
        show_result = vis_show.create_visualization(result, metrics)

        self.assertIsNone(show_result)  # No save path
        mock_show.assert_called_once()

    def test_visualization_with_various_attribution_shapes(self):
        vis = Visualiser(show=False, save_path=None)

        # Test different attribution shapes
        attribution_shapes = [
            (224, 224),  # 2D
            (1, 224, 224),  # 3D with single channel
            (3, 224, 224),  # 3D with RGB channels
            (14, 14),  # Small 2D (like GradCAM output)
        ]

        for shape in attribution_shapes:
            result = XAIExplanationResult(
                image_path=str(self.test_image_path),
                image_name="shape_test.jpg",
                model_name="TestModel",
                explainer_name="TestExplainer",
                predicted_class=1,
                true_label=1,
                prediction_correct=True,
                attribution=torch.randn(*shape),
                has_bbox=False,
                bbox=None,
            )

            with patch("matplotlib.pyplot.close"):
                vis_result = vis.create_visualization(result)

            self.assertIsNone(vis_result)  # No save path provided


if __name__ == "__main__":
    unittest.main()
