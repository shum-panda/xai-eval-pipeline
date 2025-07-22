from typing import Any, List

import torch
from torch import nn

from src.pipeline_moduls.models.base.interface.xai_model import XAIModel


class PytorchHubModel(XAIModel):
    """XAI Model for PyTorch Hub models"""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)

        # Configuration with defaults
        self.pretrained = kwargs.get("pretrained", True)
        self.repo = kwargs.get("repo", "pytorch/vision:v0.10.0")

        # Load the _model from PyTorch Hub
        self.model = self._load_from_hub(model_name, **kwargs)

        # Setup for XAI
        self._setup_for_xai()

    def _load_from_hub(self, model_name: str, **kwargs) -> torch.nn.Module:
        """Load _model from PyTorch Hub"""
        try:
            self._logger.info(
                f"Loading PyTorch Hub _model '{model_name}' "
                f"(pretrained={self.pretrained})"
            )

            model = torch.hub.load(
                self.repo, model_name, pretrained=self.pretrained, verbose=False
            )

            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load PyTorch Hub _model '{model_name}': {str(e)}"
            ) from e

    def _setup_for_xai(self) -> None:
        """Prepare _model for XAI usage (from your existing PytorchModel)"""

        # Set to eval mode (deterministic)
        self.model.eval()

        self.model = self.model.to(self._device)

        self._logger.info(f"Model '{self._model_name}' ready for XAI on {self._device}")

    def get_conv_layers(self) -> List[str]:
        """Get all convolutional layer names for XAI target selection"""
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
        """Get basic _model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        conv_layers = self.get_conv_layers()

        return {
            "name": self._model_name,
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
        """Get the underlying PyTorch _model for XAI methods"""
        return self.model
