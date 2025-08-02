import os
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn

from src.pipeline_moduls.models.base.xai_model import XAIModel


class CustomModel(XAIModel):
    """
    XAI Model wrapper for arbitrary custom PyTorch models.

    Supports either passing a model class (to instantiate) or a model instance directly.
    """

    @property
    def pytorch_model(self) -> torch.nn.Module:
        return  self.model

    def __init__(
        self,
        model_name: str,
        model_class: Optional[Type[nn.Module]] = None,
        model_instance: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the custom model.

        Args:
            model_name (str): A string identifier for the model.
            model_class (Optional[Type[nn.Module]]): A class reference to instantiate
             a model.
            model_instance (Optional[nn.Module]): A pre-initialized model instance.
            **kwargs (Any): Additional arguments passed to model_class when
                instantiating.

        Raises:
            ValueError: If neither model_class nor model_instance is provided.
        """
        super().__init__(model_name)

        if model_instance is not None:
            self.model: nn.Module = model_instance
        elif model_class is not None:
            self.model = model_class(**kwargs)
        else:
            raise ValueError("Either model_class or model_instance must be provided")

        self.config: Dict[str, Any] = kwargs

        self._setup_for_xai()

    def _setup_for_xai(self) -> None:
        """
        Prepare the model for use in XAI methods.

        Sets the model to evaluation mode and moves it to the correct device.
        """
        self.model.eval()
        self.model = self.model.to(self._device)

        self._logger.info(
            f"Custom model '{self._model_name}' ready for XAI on {self._device}"
        )

    def load_weights(self, weights_path: str) -> None:
        """
        Load model weights from a given file path.

        Args:
            weights_path (str): Path to the weights file (.pt or .pth)

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If loading the weights fails.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location="cpu")

            # Support various formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            self.model.load_state_dict(state_dict, strict=False)
            self._logger.info(f"Loaded weights from '{weights_path}'")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from '{weights_path}': {str(e)}"
            ) from e

    def get_conv_layers(self) -> List[str]:
        """
        Return the names of all convolutional layers in the model.

        Returns:
            List[str]: List of layer names for all `nn.Conv2d` modules.
        """
        return [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Conv2d)
        ]

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        """
        Retrieve a model layer by its name.

        Args:
            layer_name (str): Name of the layer to retrieve.

        Returns:
            nn.Module: The corresponding PyTorch module.

        Raises:
            ValueError: If the layer name is not found.
        """
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module

        available_layers = [name for name, _ in self.model.named_modules() if name]
        raise ValueError(
            f"Layer '{layer_name}' not found. Available layers: "
            f" {available_layers[:10]}..."
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get metadata about the model for logging or visualization.

        Returns:
            Dict[str, Any]: A dictionary with model information.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        conv_layers = self.get_conv_layers()

        return {
            "name": self._model_name,
            "type": "custom",
            "class": type(self.model).__name__,
            "total_parameters": total_params,
            "num_conv_layers": len(conv_layers),
            "device": str(next(self.model.parameters()).device),
            "sample_conv_layers": conv_layers[-3:] if conv_layers else [],
            "config": self.config,
        }
