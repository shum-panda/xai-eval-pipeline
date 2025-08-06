import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.pipeline.pipeline_moduls.models.implementation.custom_model import CustomModel
from src.pipeline.pipeline_moduls.models.implementation.pytorch_hub_model import (
    PytorchHubModel,
)
from src.pipeline.pipeline_moduls.models.model_registry import ModelRegistry
from src.pipeline.pipeline_moduls.models.xai_model_factory import XAIModelFactory


class MockPyTorchModel(nn.Module):
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


class TestPytorchHubModel(unittest.TestCase):

    @patch("torch.hub.load")
    def setUp(self, mock_hub_load):
        self.mock_model = MockPyTorchModel()
        mock_hub_load.return_value = self.mock_model
        self.hub_model = PytorchHubModel("resnet18", pretrained=True)

    def test_initialization(self):
        self.assertEqual(self.hub_model.model_name, "resnet18")
        self.assertTrue(self.hub_model.pretrained)
        self.assertEqual(self.hub_model.repo, "pytorch/vision:v0.10.0")

    @patch("torch.hub.load")
    def test_initialization_with_custom_repo(self, mock_hub_load):
        mock_hub_load.return_value = self.mock_model
        custom_repo = "custom/repo:v1.0"
        model = PytorchHubModel("resnet18", repo=custom_repo, pretrained=False)

        self.assertEqual(model.repo, custom_repo)
        self.assertFalse(model.pretrained)
        mock_hub_load.assert_called_with(
            custom_repo, "resnet18", pretrained=False, verbose=False
        )

    @patch("torch.hub.load")
    def test_load_from_hub_failure(self, mock_hub_load):
        mock_hub_load.side_effect = RuntimeError("Network error")

        with self.assertRaises(RuntimeError) as context:
            PytorchHubModel("invalid_model")

        self.assertIn("Failed to load PyTorch Hub model", str(context.exception))

    def test_get_conv_layers(self):
        conv_layers = self.hub_model.get_conv_layers()
        self.assertIn("conv1", conv_layers)
        self.assertIn("conv2", conv_layers)
        self.assertEqual(len(conv_layers), 2)

    def test_get_layer_by_name_success(self):
        layer = self.hub_model.get_layer_by_name("conv1")
        self.assertIsInstance(layer, nn.Conv2d)
        self.assertEqual(layer.in_channels, 3)
        self.assertEqual(layer.out_channels, 64)

    def test_get_layer_by_name_failure(self):
        with self.assertRaises(ValueError) as context:
            self.hub_model.get_layer_by_name("nonexistent_layer")

        self.assertIn("Layer 'nonexistent_layer' not found", str(context.exception))

    def test_get_model_info(self):
        info = self.hub_model.get_model_info()

        self.assertEqual(info["name"], "resnet18")
        self.assertEqual(info["type"], "pytorch_hub")
        self.assertEqual(info["class"], "MockPyTorchModel")
        self.assertIsInstance(info["total_parameters"], int)
        self.assertEqual(info["num_conv_layers"], 2)
        self.assertTrue(info["pretrained"])

    def test_pytorch_model_property(self):
        pytorch_model = self.hub_model.pytorch_model
        self.assertIsInstance(pytorch_model, MockPyTorchModel)

    def test_get_predictions(self):
        batch_images = torch.randn(2, 3, 224, 224)
        predictions = self.hub_model.get_predictions(batch_images)

        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape, (2, 1000))

    def test_device_handling(self):
        device = self.hub_model.device
        self.assertIsInstance(device, torch.device)


class TestCustomModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.pth"

        # Save a dummy model
        dummy_model = MockPyTorchModel()
        torch.save(dummy_model.state_dict(), self.model_path)

    @patch(
        "src.pipeline.pipeline_moduls.models.implementation.custom_model.CustomModel._create_model_architecture"
    )
    def test_initialization_from_checkpoint(self, mock_create_arch):
        mock_create_arch.return_value = MockPyTorchModel()

        model = CustomModel("test_model", model_path=str(self.model_path))
        self.assertEqual(model.model_name, "test_model")

    @patch(
        "src.pipeline.pipeline_moduls.models.implementation.custom_model.CustomModel._create_model_architecture"
    )
    def test_initialization_without_checkpoint(self, mock_create_arch):
        mock_create_arch.return_value = MockPyTorchModel()

        model = CustomModel("test_model")
        self.assertEqual(model.model_name, "test_model")
        mock_create_arch.assert_called_once()

    @patch(
        "src.pipeline.pipeline_moduls.models.implementation.custom_model.CustomModel._create_model_architecture"
    )
    def test_invalid_checkpoint_path(self, mock_create_arch):
        mock_create_arch.return_value = MockPyTorchModel()

        with self.assertRaises(FileNotFoundError):
            CustomModel("test_model", model_path="/nonexistent/path.pth")


class TestXAIModelFactory(unittest.TestCase):

    @patch(
        "src.pipeline.pipeline_moduls.models.xai_model_factory.ModelRegistry.get_model"
    )
    def test_create_pytorch_hub_model(self, mock_get_model):
        mock_get_model.return_value = PytorchHubModel

        with patch("torch.hub.load") as mock_hub_load:
            mock_hub_load.return_value = MockPyTorchModel()

            config = {
                "model_type": "pytorch_hub",
                "model_name": "resnet18",
                "pretrained": True,
            }

            model = XAIModelFactory.create_model(config)
            self.assertIsInstance(model, PytorchHubModel)
            self.assertEqual(model.model_name, "resnet18")

    @patch(
        "src.pipeline.pipeline_moduls.models.xai_model_factory.ModelRegistry.get_model"
    )
    def test_create_custom_model(self, mock_get_model):
        mock_get_model.return_value = CustomModel

        with patch(
            "src.pipeline.pipeline_moduls.models.implementation.custom_model.CustomModel._create_model_architecture"
        ) as mock_create:
            mock_create.return_value = MockPyTorchModel()

            config = {
                "model_type": "custom",
                "model_name": "my_custom_model",
                "model_path": "/path/to/model.pth",
            }

            model = XAIModelFactory.create_model(config)
            self.assertIsInstance(model, CustomModel)
            self.assertEqual(model.model_name, "my_custom_model")

    @patch(
        "src.pipeline.pipeline_moduls.models.xai_model_factory.ModelRegistry.get_model"
    )
    def test_unsupported_model_type(self, mock_get_model):
        mock_get_model.side_effect = KeyError("Model type not registered")

        config = {"model_type": "unsupported_type", "model_name": "test_model"}

        with self.assertRaises(KeyError):
            XAIModelFactory.create_model(config)


class TestModelRegistry(unittest.TestCase):

    def test_pytorch_hub_registration(self):
        model_class = ModelRegistry.get_model("pytorch_hub")
        self.assertEqual(model_class, PytorchHubModel)

    def test_custom_model_registration(self):
        model_class = ModelRegistry.get_model("custom")
        self.assertEqual(model_class, CustomModel)

    def test_invalid_model_type(self):
        with self.assertRaises(KeyError):
            ModelRegistry.get_model("nonexistent_type")

    def test_list_available_models(self):
        available = ModelRegistry.list_available()
        self.assertIn("pytorch_hub", available)
        self.assertIn("custom", available)


class TestModelIntegration(unittest.TestCase):

    @patch("torch.hub.load")
    def test_full_model_pipeline(self, mock_hub_load):
        mock_hub_load.return_value = MockPyTorchModel()

        # Create model through factory
        config = {
            "model_type": "pytorch_hub",
            "model_name": "resnet18",
            "pretrained": True,
        }

        with patch(
            "src.pipeline.pipeline_moduls.models.xai_model_factory.ModelRegistry.get_model"
        ) as mock_get_model:
            mock_get_model.return_value = PytorchHubModel
            model = XAIModelFactory.create_model(config)

            # Test basic functionality
            self.assertEqual(model.model_name, "resnet18")

            # Test inference
            test_input = torch.randn(1, 3, 224, 224)
            output = model.get_predictions(test_input)
            self.assertEqual(output.shape, (1, 1000))

            # Test layer access
            conv_layers = model.get_conv_layers()
            self.assertGreater(len(conv_layers), 0)

            # Test layer retrieval
            first_conv = model.get_layer_by_name(conv_layers[0])
            self.assertIsInstance(first_conv, nn.Conv2d)

            # Test model info
            info = model.get_model_info()
            self.assertIn("name", info)
            self.assertIn("total_parameters", info)

    def test_model_device_consistency(self):
        with patch("torch.hub.load") as mock_hub_load:
            mock_model = MockPyTorchModel()
            mock_hub_load.return_value = mock_model

            hub_model = PytorchHubModel("resnet18")

            # Test that model and its parameters are on the same device
            model_device = next(hub_model.pytorch_model.parameters()).device
            self.assertEqual(hub_model.device, model_device)

    def test_model_evaluation_mode(self):
        with patch("torch.hub.load") as mock_hub_load:
            mock_model = MockPyTorchModel()
            mock_hub_load.return_value = mock_model

            hub_model = PytorchHubModel("resnet18")

            # Test that model is in evaluation mode
            self.assertFalse(hub_model.pytorch_model.training)


if __name__ == "__main__":
    unittest.main()
