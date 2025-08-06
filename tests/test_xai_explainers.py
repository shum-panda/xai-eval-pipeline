import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from src.pipeline.pipeline_moduls.xai_methods.impl.grad_cam.grand_cam_explainer import (
    GradCamExplainer,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.guided_backprop.guided_backprop_expl import (
    GuidedBackpropExplainer,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.integrated_gradients.integrated_gradients_explainer import (
    IntegratedGradientsExplainer,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.score_cam.score_cam_explainer import (
    ScoreCamExplainer,
)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MockXAIModel:
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def get_predictions(self, images):
        return self.pytorch_model(images)

    def get_layer_by_name(self, name):
        return dict(self.pytorch_model.named_modules())[name]


class TestGradCamExplainer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockModel()
        self.xai_model = MockXAIModel(self.mock_model)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        self.target_labels = torch.tensor([1, 5])

    def test_initialization_with_defaults(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)
        self.assertIsNotNone(explainer.grad_cam)
        self.assertEqual(explainer.get_name(), "grad_cam")

    def test_initialization_with_custom_config(self):
        config = {
            "target_layer": "conv2",
            "relu_attributions": False,
            "interpolate_mode": "nearest",
        }
        explainer = GradCamExplainer(self.xai_model, use_defaults=False, **config)
        self.assertEqual(explainer.target_layer, "conv2")
        self.assertFalse(explainer.relu_attributions)
        self.assertEqual(explainer.interpolate_mode, "nearest")

    def test_target_layer_selection_by_name(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)
        layer_name = explainer._select_target_layer("conv1")
        self.assertEqual(layer_name, "conv1")

    def test_target_layer_selection_by_index(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)
        layer_name = explainer._select_target_layer(-1)
        self.assertEqual(layer_name, "conv2")

    def test_compute_attributions(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)
        attributions = explainer._compute_attributions(
            self.input_tensor, self.target_labels
        )

        self.assertIsInstance(attributions, torch.Tensor)
        self.assertEqual(attributions.shape[0], self.batch_size)
        self.assertEqual(attributions.shape[-2:], self.input_tensor.shape[-2:])

    def test_explain_method(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)
        result = explainer.explain(self.input_tensor, self.target_labels, top_k=3)

        self.assertIsNotNone(result.attributions)
        self.assertIsNotNone(result.predictions)
        self.assertIsNotNone(result.confidence)
        self.assertEqual(result.topk_predictions.shape[1], 3)

    def test_invalid_target_layer(self):
        with self.assertRaises(ValueError):
            explainer = GradCamExplainer(self.xai_model, use_defaults=True)
            explainer._select_target_layer("nonexistent_layer")

    def test_parameters_property(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)
        params = explainer.parameters
        self.assertIn("target_layer", params)
        self.assertIn("relu_attributions", params)
        self.assertIn("interpolate_mode", params)


class TestGuidedBackpropExplainer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockModel()
        self.xai_model = MockXAIModel(self.mock_model)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        self.target_labels = torch.tensor([1, 5])

    @patch(
        "src.pipeline.pipeline_moduls.xai_methods.impl.guided_backprop.guided_backprop_expl.GuidedBackprop"
    )
    def test_initialization(self, mock_guided_backprop):
        mock_instance = Mock()
        mock_guided_backprop.return_value = mock_instance

        explainer = GuidedBackpropExplainer(self.xai_model, use_defaults=True)
        self.assertEqual(explainer.get_name(), "guided_backprop")

    @patch(
        "src.pipeline.pipeline_moduls.xai_methods.impl.guided_backprop.guided_backprop_expl.GuidedBackprop"
    )
    def test_compute_attributions(self, mock_guided_backprop):
        mock_instance = Mock()
        mock_instance.attribute.return_value = torch.randn(self.batch_size, 3, 224, 224)
        mock_guided_backprop.return_value = mock_instance

        explainer = GuidedBackpropExplainer(self.xai_model, use_defaults=True)
        attributions = explainer._compute_attributions(
            self.input_tensor, self.target_labels
        )

        self.assertIsInstance(attributions, torch.Tensor)
        self.assertEqual(attributions.shape, (self.batch_size, 3, 224, 224))


class TestIntegratedGradientsExplainer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockModel()
        self.xai_model = MockXAIModel(self.mock_model)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        self.target_labels = torch.tensor([1, 5])

    @patch(
        "src.pipeline.pipeline_moduls.xai_methods.impl.integrated_gradients.integrated_gradients_explainer.IntegratedGradients"
    )
    def test_initialization(self, mock_integrated_gradients):
        mock_instance = Mock()
        mock_integrated_gradients.return_value = mock_instance

        explainer = IntegratedGradientsExplainer(self.xai_model, use_defaults=True)
        self.assertEqual(explainer.get_name(), "integrated_gradients")

    @patch(
        "src.pipeline.pipeline_moduls.xai_methods.impl.integrated_gradients.integrated_gradients_explainer.IntegratedGradients"
    )
    def test_compute_attributions(self, mock_integrated_gradients):
        mock_instance = Mock()
        mock_instance.attribute.return_value = torch.randn(self.batch_size, 3, 224, 224)
        mock_integrated_gradients.return_value = mock_instance

        explainer = IntegratedGradientsExplainer(self.xai_model, use_defaults=True)
        attributions = explainer._compute_attributions(
            self.input_tensor, self.target_labels
        )

        self.assertIsInstance(attributions, torch.Tensor)
        self.assertEqual(attributions.shape, self.input_tensor.shape)


class TestScoreCamExplainer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockModel()
        self.xai_model = MockXAIModel(self.mock_model)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        self.target_labels = torch.tensor([1, 5])

    def test_initialization(self):
        explainer = ScoreCamExplainer(self.xai_model, use_defaults=True)
        self.assertEqual(explainer.get_name(), "score_cam")
        self.assertIsNotNone(explainer.target_layer)

    def test_target_layer_selection(self):
        explainer = ScoreCamExplainer(self.xai_model, use_defaults=True)
        layer_name = explainer._select_target_layer(-1)  # Last conv layer
        self.assertEqual(layer_name, "conv2")

    def test_hook_registration_and_removal(self):
        explainer = ScoreCamExplainer(self.xai_model, use_defaults=True)

        # Test hook registration
        explainer._register_hook()
        self.assertIsNotNone(explainer.hook_handle)

        # Test hook removal
        explainer._remove_hook()
        self.assertIsNone(explainer.hook_handle)

    @patch.object(MockXAIModel, "get_predictions")
    def test_compute_attributions(self, mock_get_predictions):
        # Mock predictions to return proper scores
        mock_get_predictions.return_value = torch.randn(self.batch_size, 1000)

        explainer = ScoreCamExplainer(self.xai_model, use_defaults=True)

        # Mock feature map that would be captured by the hook
        explainer.feature_map = torch.randn(
            self.batch_size, 512, 7, 7
        )  # Typical feature map size

        attributions = explainer._compute_attributions(
            self.input_tensor, self.target_labels
        )

        self.assertIsInstance(attributions, torch.Tensor)
        self.assertEqual(attributions.shape, (self.batch_size, 224, 224))

        # Check that attributions are normalized (max should be 1 or close to 1)
        max_vals = attributions.amax(dim=(-2, -1))
        self.assertTrue(torch.all(max_vals <= 1.0 + 1e-6))

    def test_parameters_property(self):
        explainer = ScoreCamExplainer(self.xai_model, use_defaults=True)
        params = explainer.parameters
        self.assertIn("target_layer", params)
        self.assertIsInstance(params["target_layer"], str)


class TestExplainerIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockModel()
        self.xai_model = MockXAIModel(self.mock_model)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        self.target_labels = torch.tensor([1, 5])

    def test_gradcam_end_to_end(self):
        explainer = GradCamExplainer(self.xai_model, use_defaults=True)

        # Test full explanation pipeline
        result = explainer.explain(self.input_tensor, self.target_labels, top_k=5)

        # Verify result structure
        self.assertIsNotNone(result.attributions)
        self.assertIsNotNone(result.predictions)
        self.assertIsNotNone(result.confidence)
        self.assertEqual(result.predictions.shape[0], self.batch_size)
        self.assertEqual(result.topk_predictions.shape, (self.batch_size, 5))

        # Verify attribution dimensions
        self.assertEqual(result.attributions.shape[0], self.batch_size)
        self.assertEqual(result.attributions.shape[-2:], self.input_tensor.shape[-2:])

    def test_multiple_explainers_consistency(self):
        explainers = [GradCamExplainer(self.xai_model, use_defaults=True)]

        results = []
        for explainer in explainers:
            result = explainer.explain(self.input_tensor, self.target_labels, top_k=3)
            results.append(result)

        # All explainers should produce results with consistent shapes
        for result in results:
            self.assertEqual(result.predictions.shape[0], self.batch_size)
            self.assertEqual(result.attributions.shape[0], self.batch_size)


if __name__ == "__main__":
    unittest.main()
