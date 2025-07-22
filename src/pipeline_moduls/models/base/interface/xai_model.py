import logging
from abc import ABC, abstractmethod
from typing import Any, List

import torch


class XAIModel(ABC):
    """Abstract base class for models used in XAI applications"""

    def __init__(self, model_name: str):
        self._model_name: str = model_name
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def model_name(self):
        return self._model_name

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def get_conv_layers(self) -> List[str]:
        """Get all convolutional layer names for XAI target selection

        Returns:
            List of layer names that are Conv2d layers
        """
        pass

    @abstractmethod
    def get_layer_by_name(self, layer_name: str) -> Any:
        """Get a specific layer by name for XAI target layers

        Args:
            layer_name: Name of the layer to retrieve

        Returns:
            The layer module

        Raises:
            ValueError: If layer name not found
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get basic _model information

        Returns:
            Dictionary with _model metadata (name, type, parameters, etc.)
        """
        pass

    @abstractmethod
    def get_pytorch_model(self) -> torch.nn.Module:
        """Get the underlying PyTorch _model for XAI methods

        Returns:
            The raw PyTorch nn.Module
        """
        pass
