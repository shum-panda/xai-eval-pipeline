from typing import Any, List

import torch

from pipeline_moduls.models.base.interface.xai_model import XAIModel


class PytorchHubModel(XAIModel):
    """XAI Model for PyTorch Hub models"""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)

        # Configuration with defaults
        self.pretrained = kwargs.get("pretrained", True)
        self.repo = kwargs.get("repo", "pytorch/vision:v0.10.0")

        # Load the model from PyTorch Hub
        self.model = self._load_from_hub(model_name, **kwargs)

        # Setup for XAI
        self._setup_for_xai()

    def _load_from_hub(self, model_name: str, **kwargs) -> torch.nn.Module:
        """Load model from PyTorch Hub"""
        try:
            self.logger.info(
                f"Loading PyTorch Hub model '{model_name}' (pretrained={self.pretrained})"
            )

            model = torch.hub.load(
                self.repo, model_name, pretrained=self.pretrained, verbose=False
            )

            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load PyTorch Hub model '{model_name}': {str(e)}"
            ) from e

    def _setup_for_xai(self) -> None:
        """Prepare model for XAI usage (from your existing PytorchModel)"""
        import torch

        # Set to eval mode (deterministic)
        self.model.eval()

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.logger.info(f"Model '{self.model_name}' ready for XAI on {device}")

    def get_conv_layers(self) -> List[str]:
        """Get all convolutional layer names for XAI target selection"""
        import torch.nn as nn

        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers

    def get_layer_by_name(self, layer_name: str) -> Any:
        """Get a specific layer by name for XAI target layers"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module

        available_layers = [name for name, _ in self.model.named_modules() if name]
        raise ValueError(
            f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}"
            f"..."
        )

    def get_model_info(self) -> dict:
        """Get basic model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        conv_layers = self.get_conv_layers()

        return {
            "name": self.model_name,
            "type": "pytorch_hub",
            "class": type(self.model).__name__,
            "total_parameters": total_params,
            "num_conv_layers": len(conv_layers),
            "device": str(next(self.model.parameters()).device),
            "sample_conv_layers": (
                conv_layers[-3:] if conv_layers else []
            ),  # Last 3 for GradCAM
            "repo": self.repo,
            "pretrained": self.pretrained,
        }

    def get_pytorch_model(self) -> torch.nn.Module:
        """Get the underlying PyTorch model for XAI methods"""
        return self.model
