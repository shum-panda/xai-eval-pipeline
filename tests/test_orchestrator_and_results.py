import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch
import yaml

from src.pipeline.control.orchestrator import Orchestrator
from src.pipeline.control.utils import MasterConfig
from src.pipeline.control.utils.dataclasses.xai_explanation_result import (
    XAIExplanationResult,
)
from src.pipeline.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults
from src.pipeline.pipeline_moduls.resultmanager.result_manager import ResultManager


class TestResultManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.result_manager = ResultManager(attribution_dir=str(self.temp_dir))

        # Create mock XAI results
        self.mock_results = [
            XAIExplanationResult(
                image_path="/path/to/image1.jpg",
                image_name="image1.jpg",
                model_name="ResNet50",
                explainer_name="GradCAM",
                predicted_class=5,
                true_label=5,
                prediction_correct=True,
                attribution=torch.randn(224, 224),
                has_bbox=True,
                bbox=torch.tensor([[50, 50, 150, 150]]),
                processing_time=0.1,
                timestamp="1234567890",
            ),
            XAIExplanationResult(
                image_path="/path/to/image2.jpg",
                image_name="image2.jpg",
                model_name="ResNet50",
                explainer_name="GradCAM",
                predicted_class=3,
                true_label=7,
                prediction_correct=False,
                attribution=torch.randn(224, 224),
                has_bbox=False,
                bbox=None,
                processing_time=0.15,
                timestamp="1234567891",
            ),
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        self.assertEqual(self.result_manager.results_count, 0)
        self.assertEqual(len(self.result_manager.results), 0)
        self.assertTrue(Path(self.temp_dir).exists())

    def test_add_results(self):
        self.result_manager.add_results(self.mock_results)

        self.assertEqual(self.result_manager.results_count, 2)
        self.assertEqual(len(self.result_manager.results), 2)

    def test_results_property_returns_copy(self):
        self.result_manager.add_results(self.mock_results)
        results_copy = self.result_manager.results

        # Modify the copy
        results_copy.clear()

        # Original should be unchanged
        self.assertEqual(self.result_manager.results_count, 2)

    def test_build_dataframe_empty_results(self):
        with self.assertRaises(ValueError):
            self.result_manager.build_dataframe()

    def test_build_dataframe_with_results(self):
        self.result_manager.add_results(self.mock_results)
        df = self.result_manager.build_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("image_name", df.columns)
        self.assertIn("model_name", df.columns)
        self.assertIn("predicted_class", df.columns)
        self.assertIn("prediction_correct", df.columns)

    def test_dataframe_property_builds_if_none(self):
        self.result_manager.add_results(self.mock_results)

        # Access property for first time
        df = self.result_manager.dataframe
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_convert_result_to_csv_dict(self):
        result = self.mock_results[0]
        csv_dict = self.result_manager._convert_result_to_csv_dict(result)

        # Should not contain large tensors
        self.assertNotIn("image", csv_dict)
        self.assertNotIn("attribution", csv_dict)
        self.assertNotIn("bbox", csv_dict)

        # Should contain basic metadata
        self.assertIn("image_name", csv_dict)
        self.assertIn("predicted_class", csv_dict)
        self.assertIn("prediction_correct", csv_dict)

    def test_tensor_to_string_representation_small_tensor(self):
        small_tensor = torch.tensor([1, 2, 3])
        result = self.result_manager._tensor_to_string_representation(small_tensor)

        self.assertIn("Tensor", result)
        self.assertIn("shape", result)
        self.assertIn("values", result)

    def test_tensor_to_string_representation_large_tensor(self):
        large_tensor = torch.randn(100, 100)
        result = self.result_manager._tensor_to_string_representation(large_tensor)

        self.assertIn("Tensor", result)
        self.assertIn("shape", result)
        self.assertIn("dtype", result)
        self.assertNotIn("values", result)  # Too large to show values

    def test_tensor_to_string_representation_none(self):
        with self.assertRaises(ValueError):
            self.result_manager._tensor_to_string_representation(None)

    def test_save_dataframe(self):
        self.result_manager.add_results(self.mock_results)
        save_path = Path(self.temp_dir) / "test_results.csv"

        self.result_manager.save_dataframe(str(save_path))

        self.assertTrue(save_path.exists())

        # Verify CSV can be read back
        df = pd.read_csv(save_path)
        self.assertEqual(len(df), 2)

    def test_save_dataframe_with_metrics(self):
        self.result_manager.add_results(self.mock_results)

        # Create mock metrics
        mock_metrics = [
            MetricResults(
                sample_id="image1",
                values={
                    "IoU": 0.8,
                    "point_game": 1.0,
                    "PixelPrecisionRecall": {"precision": 0.75, "recall": 0.85},
                },
            ),
            MetricResults(
                sample_id="image2",
                values={
                    "IoU": 0.6,
                    "point_game": 0.0,
                    "PixelPrecisionRecall": {"precision": 0.65, "recall": 0.70},
                },
            ),
        ]

        output_path = self.result_manager.save_dataframe_with_metrics(
            Path(self.temp_dir), individual_metrics=mock_metrics
        )

        self.assertTrue(output_path.exists())

        # Verify metrics were added
        df = pd.read_csv(output_path)
        self.assertIn("IoU", df.columns)
        self.assertIn("point_game", df.columns)
        self.assertIn("PixelPrecisionRecall_precision", df.columns)
        self.assertIn("PixelPrecisionRecall_recall", df.columns)
        self.assertEqual(df.loc[0, "IoU"], 0.8)
        self.assertEqual(df.loc[1, "point_game"], 0.0)

    def test_add_individual_metrics_to_df(self):
        self.result_manager.add_results(self.mock_results)
        df = self.result_manager.build_dataframe()

        mock_metrics = [
            MetricResults(
                sample_id="test1", values={"metric1": 0.5, "nested": {"a": 1, "b": 2}}
            ),
            MetricResults(
                sample_id="test2", values={"metric1": 0.7, "nested": {"a": 3, "b": 4}}
            ),
        ]

        enhanced_df = self.result_manager._add_individual_metrics_to_df(
            df, mock_metrics
        )

        self.assertIn("metric1", enhanced_df.columns)
        self.assertIn("nested_a", enhanced_df.columns)
        self.assertIn("nested_b", enhanced_df.columns)
        self.assertIn("evaluation_metrics_json", enhanced_df.columns)

    def test_save_evaluation_summary_to_file(self):
        mock_summary = EvaluationSummary(
            total_samples=10,
            prediction_accuracy=0.8,
            samples_with_bbox=5,
            average_processing_time=0.12,
            metric_averages={"IoU": 0.75, "point_game": 0.6},
        )

        output_path = self.result_manager.save_evaluation_summary_to_file(
            mock_summary, Path(self.temp_dir)
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(output_path.name.endswith(".yaml"))

        # Verify YAML content
        with open(output_path, "r") as f:
            loaded_data = yaml.safe_load(f)

        self.assertEqual(loaded_data["total_samples"], 10)
        self.assertEqual(loaded_data["prediction_accuracy"], 0.8)

    def test_reset(self):
        self.result_manager.add_results(self.mock_results)
        self.assertEqual(self.result_manager.results_count, 2)

        self.result_manager.reset()

        self.assertEqual(self.result_manager.results_count, 0)
        self.assertEqual(len(self.result_manager.results), 0)

    def test_get_latest_results(self):
        self.result_manager.add_results(self.mock_results)

        # Get latest 1 result
        latest = self.result_manager.get_latest_results(1)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].image_name, "image2.jpg")  # Last added

        # Get more results than available
        latest_all = self.result_manager.get_latest_results(10)
        self.assertEqual(len(latest_all), 2)

    def test_clean_remaining_tensors(self):
        data_with_tensors = {
            "tensor_field": torch.tensor([1, 2, 3]),
            "normal_field": "test",
            "nested_dict": {"inner_tensor": torch.tensor([4, 5]), "inner_normal": 42},
            "tensor_list": [torch.tensor([6]), "string", 7],
        }

        cleaned = self.result_manager._clean_remaining_tensors(data_with_tensors)

        self.assertIsInstance(cleaned["tensor_field"], str)
        self.assertEqual(cleaned["normal_field"], "test")
        self.assertIsInstance(cleaned["nested_dict"]["inner_tensor"], str)
        self.assertEqual(cleaned["nested_dict"]["inner_normal"], 42)
        self.assertIsInstance(cleaned["tensor_list"][0], str)
        self.assertEqual(cleaned["tensor_list"][1], "string")


class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create mock config
        self.mock_config = Mock(spec=MasterConfig)
        self.mock_config.experiment.name = "test_experiment"
        self.mock_config.experiment.output_dir = str(self.temp_dir)
        self.mock_config.experiment.top_k = 5
        self.mock_config.model.name = "resnet18"
        self.mock_config.model.transform = False
        self.mock_config.xai.name = "grad_cam"
        self.mock_config.xai.kwargs = {}
        self.mock_config.xai.use_defaults = True
        self.mock_config.data.batch_size = 2
        self.mock_config.data.num_workers = 0
        self.mock_config.data.pin_memory = False
        self.mock_config.data.shuffle = False
        self.mock_config.data.resize = [224, 224]
        self.mock_config.data.max_batches = 1
        self.mock_config.visualization.show = False
        self.mock_config.visualization.save = False
        self.mock_config.visualization.max_visualizations = 1
        self.mock_config.metric.kwargs = {}
        self.mock_config.logging.level = "INFO"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    def test_orchestrator_initialization(
        self, mock_mapper, mock_xai_factory, mock_model_factory, mock_setup_logger
    ):
        # Setup mocks
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        mock_xai_factory.return_value.list_available_explainers.return_value = [
            "grad_cam"
        ]

        orchestrator = Orchestrator(self.mock_config)

        self.assertEqual(orchestrator._pipeline_status, "initialized")
        self.assertEqual(orchestrator._current_step, "none")
        self.assertIsNone(orchestrator._pipeline_error)

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    def test_pipeline_status_property(
        self, mock_mapper, mock_xai_factory, mock_model_factory, mock_setup_logger
    ):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        orchestrator = Orchestrator(self.mock_config)
        status = orchestrator.pipeline_status

        self.assertIn("status", status)
        self.assertIn("current_step", status)
        self.assertIn("has_error", status)
        self.assertIn("error_details", status)
        self.assertIn("mlflow_active", status)

        self.assertEqual(status["status"], "initialized")
        self.assertEqual(status["current_step"], "none")
        self.assertFalse(status["has_error"])

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    @patch("src.pipeline.control.orchestrator.mlflow")
    @patch("src.pipeline.control.orchestrator.create_dataloader")
    def test_prepare_experiment(
        self,
        mock_create_dataloader,
        mock_mlflow,
        mock_mapper,
        mock_xai_factory,
        mock_model_factory,
        mock_setup_logger,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.pytorch_model = Mock()
        mock_model_factory.return_value.create.return_value = mock_model

        mock_mlflow.active_run.return_value = None
        mock_run = Mock()
        mock_mlflow.start_run.return_value = mock_run

        orchestrator = Orchestrator(self.mock_config)
        orchestrator.prepare_experiment()

        mock_mlflow.start_run.assert_called_once_with(run_name="test_experiment")
        mock_mlflow.pytorch.log_model.assert_called_once()
        mock_mlflow.log_param.assert_called()

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    @patch("src.pipeline.control.orchestrator.create_dataloader")
    def test_setup_dataloader(
        self,
        mock_create_dataloader,
        mock_mapper,
        mock_xai_factory,
        mock_model_factory,
        mock_setup_logger,
    ):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        mock_dataloader = Mock()
        mock_dataloader.dataset = Mock()
        mock_dataloader.dataset.__len__ = Mock(return_value=100)
        mock_dataloader.__len__ = Mock(return_value=50)
        mock_create_dataloader.return_value = mock_dataloader

        orchestrator = Orchestrator(self.mock_config)
        result = orchestrator.setup_dataloader(
            project_root=None,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            target_size=[224, 224],
            transform=None,
        )

        self.assertEqual(result, mock_dataloader)
        mock_create_dataloader.assert_called_once()

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    def test_create_explainer_success(
        self, mock_mapper, mock_xai_factory, mock_model_factory, mock_setup_logger
    ):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        mock_explainer = Mock()
        mock_xai_factory.return_value.create_explainer.return_value = mock_explainer

        orchestrator = Orchestrator(self.mock_config)
        result = orchestrator.create_explainer("grad_cam", {}, True)

        self.assertEqual(result, mock_explainer)
        mock_xai_factory.return_value.create_explainer.assert_called_once_with(
            name="grad_cam", model=mock_model, use_defaults=True
        )

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    def test_create_explainer_failure(
        self, mock_mapper, mock_xai_factory, mock_model_factory, mock_setup_logger
    ):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        mock_xai_factory.return_value.create_explainer.side_effect = TypeError(
            "Invalid parameters"
        )

        orchestrator = Orchestrator(self.mock_config)

        with self.assertRaises(TypeError):
            orchestrator.create_explainer("grad_cam", {}, True)

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    @patch("src.pipeline.control.orchestrator.time.time")
    def test_explain_batch(
        self,
        mock_time,
        mock_mapper,
        mock_xai_factory,
        mock_model_factory,
        mock_setup_logger,
    ):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        mock_time.return_value = 1000.0

        # Create mock batch
        mock_batch = Mock()
        mock_batch.images_tensor = torch.randn(2, 3, 224, 224)
        mock_batch.labels_tensor = torch.tensor([1, 5])
        mock_batch.image_names = ["image1.jpg", "image2.jpg"]
        mock_batch.image_paths = ["/path/image1.jpg", "/path/image2.jpg"]
        mock_batch.boxes_list = [torch.empty(0, 4), torch.tensor([[10, 10, 50, 50]])]

        # Create mock explainer result
        mock_explainer_result = Mock()
        mock_explainer_result.attributions = torch.randn(2, 224, 224)
        mock_explainer_result.predictions = torch.tensor([1, 5])
        mock_explainer_result.confidence = torch.tensor([0.9, 0.8])
        mock_explainer_result.target_labels = torch.tensor([1, 5])
        mock_explainer_result.topk_predictions = torch.tensor(
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
        )
        mock_explainer_result.topk_confidences = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.5], [0.8, 0.7, 0.6, 0.5, 0.4]]
        )

        mock_explainer = Mock()
        mock_explainer.explain.return_value = mock_explainer_result
        mock_explainer.get_name.return_value = "grad_cam"
        mock_explainer.parameters = {"param1": "value1"}

        orchestrator = Orchestrator(self.mock_config)
        results = orchestrator.explain_batch(mock_batch, mock_explainer)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], XAIExplanationResult)
        self.assertEqual(results[0].image_name, "image1.jpg")
        self.assertEqual(results[1].image_name, "image2.jpg")


class TestOrchestratorIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("src.pipeline.control.orchestrator.setup_logger")
    @patch("src.pipeline.control.orchestrator.XAIModelFactory")
    @patch("src.pipeline.control.orchestrator.XAIFactory")
    @patch("src.pipeline.control.orchestrator.ImageNetLabelMapper")
    @patch("src.pipeline.control.orchestrator.create_dataloader")
    @patch("src.pipeline.control.orchestrator.mlflow")
    def test_integration_result_manager_with_orchestrator(
        self,
        mock_mlflow,
        mock_create_dataloader,
        mock_mapper,
        mock_xai_factory,
        mock_model_factory,
        mock_setup_logger,
    ):
        """Test that ResultManager properly integrates with Orchestrator"""

        # Setup config
        mock_config = Mock(spec=MasterConfig)
        mock_config.experiment.name = "integration_test"
        mock_config.experiment.output_dir = str(self.temp_dir)
        mock_config.model.name = "resnet18"
        mock_config.xai.name = "grad_cam"
        mock_config.data.batch_size = 1
        mock_config.logging.level = "INFO"

        # Mock necessary components
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model_factory.return_value.create.return_value = mock_model

        # Create orchestrator
        orchestrator = Orchestrator(mock_config)

        # Test that result manager is properly initialized
        self.assertIsInstance(orchestrator._result_manager, ResultManager)
        self.assertEqual(orchestrator._result_manager.results_count, 0)

        # Test adding results
        mock_results = [
            XAIExplanationResult(
                image_path="/test/path.jpg",
                image_name="test.jpg",
                model_name="test_model",
                explainer_name="test_explainer",
                predicted_class=1,
                true_label=1,
                prediction_correct=True,
                attribution=torch.randn(10, 10),
                processing_time=0.1,
                timestamp="123456",
            )
        ]

        orchestrator._result_manager.add_results(mock_results)
        self.assertEqual(orchestrator._result_manager.results_count, 1)

        # Test dataframe creation
        df = orchestrator._result_manager.dataframe
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
