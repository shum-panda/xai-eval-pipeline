from typing import Any, List, Type

import torch

from pipeline_moduls.models.base.interface.xai_model import XAIModel


class CustomModel(XAIModel):
    """XAI Model for custom PyTorch models"""

    def __init__(
        self,
        model_name: str,
        model_class: Type = None,
        model_instance: Any = None,
        **kwargs,
    ):
        super().__init__(model_name)

        if model_instance is not None:
            # Use provided model instance
            self.model = model_instance
        elif model_class is not None:
            # Create model from class
            self.model = model_class(**kwargs)
        else:
            raise ValueError("Either model_class or model_instance must be provided")

        self.config = kwargs

        # Setup for XAI
        self._setup_for_xai()

    def _setup_for_xai(self) -> None:
        """Prepare model for XAI usage"""
        import torch

        # Set to eval mode (deterministic)
        self.model.eval()

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.logger.info(f"Custom model '{self.model_name}' ready for XAI on {device}")

    def load_weights(self, weights_path: str) -> None:
        """Load weights into the model"""
        import os

        import torch

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Load state dict
            state_dict = torch.load(weights_path, map_location="cpu")

            # Handle different save formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Load into model
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded weights from '{weights_path}'")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from '{weights_path}': {str(e)}"
            ) from e

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
            f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}..."
        )

    def get_model_info(self) -> dict:
        """Get basic model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        conv_layers = self.get_conv_layers()

        return {
            "name": self.model_name,
            "type": "custom",
            "class": type(self.model).__name__,
            "total_parameters": total_params,
            "num_conv_layers": len(conv_layers),
            "device": str(next(self.model.parameters()).device),
            "sample_conv_layers": (
                conv_layers[-3:] if conv_layers else []
            ),  # Last 3 for GradCAM
            "config": self.config,
        }

    def get_pytorch_model(self) -> torch.nn.Module:
        """Get the underlying PyTorch model for XAI methods"""
        return self.model
