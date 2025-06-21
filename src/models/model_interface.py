import logging
from typing import List

import torch
from torch import nn


class ModelInterface:
    """Clean interface for models - only what XAI needs"""

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Setup model for XAI
        self._setup_for_xai()

    def _setup_for_xai(self):
        """Prepare model for XAI usage"""
        # Set to eval mode (deterministic)
        self.model.eval()

        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        self.logger.info(f"Model '{self.model_name}' ready for XAI on {device}")

    def get_pytorch_model(self) -> nn.Module:
        """Get the underlying PyTorch model for XAI methods"""
        return self.model

    def get_conv_layers(self) -> List[str]:
        """
        Get all convolutional layer names for XAI target selection

        Returns:
            List of layer names that are Conv2d layers
        """
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        """
        Get a specific layer by name for XAI target layers

        Args:
            layer_name: Name of the layer to retrieve

        Returns:
            The layer module

        Raises:
            ValueError: If layer name not found
        """
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module

        available_layers = [name for name, _ in self.model.named_modules() if name]
        raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}...")

    def get_model_info(self) -> dict:
        """Get basic model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        conv_layers = self.get_conv_layers()

        return {
            'name': self.model_name,
            'type': type(self.model).__name__,
            'total_parameters': total_params,
            'num_conv_layers': len(conv_layers),
            'device': str(next(self.model.parameters()).device),
            'sample_conv_layers': conv_layers[-3:] if conv_layers else []  # Last 3 for GradCAM
        }