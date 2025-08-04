import hashlib
from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from src.pipeline_moduls.models.base.xai_model import XAIModel


class PytorchHubModel(XAIModel):
    """
    XAI Model wrapper for PyTorch Hub models.

    Loads pretrained models from PyTorch Hub and prepares them for
    explainability methods by exposing utility functions such as
    retrieving convolutional layers.
    """

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the PyTorch Hub _model wrapper.

        Args:
            model_name (str): Name of the _model to load from PyTorch Hub.
            **kwargs: Additional optional parameters such as:
                - pretrained (bool): Whether to load pretrained weights (default: True).
                - repo (str): The PyTorch Hub repository reference (default:
                'pytorch/vision:v0.10.0').
                - seed (int): Random seed for deterministic behavior (default: 42).
        """
        super().__init__(model_name)

        self.pretrained: bool = kwargs.get("pretrained", True)
        self.repo: str = kwargs.get("repo", "pytorch/vision:v0.10.0")
        self.seed: int = kwargs.get("seed", 42)

        # Load the _model from PyTorch Hub
        self._model: nn.Module = self._load_from_hub(model_name, **kwargs)

        # Prepare _model for XAI usage
        self._setup_for_xai()

    def _load_from_hub(self, model_name: str, **kwargs: Any) -> nn.Module:
        """
        Load a PyTorch model from the PyTorch Hub with deterministic weights.

        Args:
            model_name (str): Name of the model to load.
            **kwargs: Additional arguments (ignored here but accepted).

        Returns:
            torch.nn.Module: The loaded PyTorch model.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        try:
            self._logger.info(
                f"Loading PyTorch Hub model '{model_name}' "
                f"(pretrained={self.pretrained}, repo={self.repo}, seed={self.seed})"
            )
            
            # Ensure deterministic behavior with configurable seed
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            
            # Clear hub cache (compatible with newer PyTorch versions)
            try:
                # For newer PyTorch versions, use get_dir() method
                import os
                cache_dir = torch.hub.get_dir()
                self._logger.debug(f"Hub cache directory: {cache_dir}")
            except Exception as e:
                self._logger.debug(f"Could not access hub cache: {e}")
            
            model = torch.hub.load(  # type: ignore
                self.repo, 
                model_name, 
                pretrained=self.pretrained, 
                verbose=False,
                force_reload=False,  # Use cached model structure but fresh weights
                trust_repo=True
            )
            
            # Log model hash for verification
            model_hash = self._compute_model_hash(model)
            self._logger.info(f"Loaded model '{model_name}' with hash: {model_hash}")
            
            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load PyTorch Hub model '{model_name}': {str(e)}"
            ) from e

    def _setup_for_xai(self) -> None:
        """
        Prepare the model for explainability (XAI) usage.

        Sets the model to evaluation mode and moves it to the configured device.
        """
        self._model.eval()
        self._model = self._model.to(self._device)
        self._logger.info(f"Model '{self._model_name}' ready for XAI on {self._device}")

    def get_conv_layers(self) -> List[str]:
        """
        Retrieve names of all convolutional layers in the model.

        Returns:
            List[str]: List of convolutional layer names, useful for
            selecting target layers in XAI methods.
        """
        conv_layers = [
            name
            for name, module in self._model.named_modules()
            if isinstance(module, nn.Conv2d)
        ]
        return conv_layers

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        """
        Retrieve a specific layer by its name.

        Args:
            layer_name (str): The name of the layer to retrieve.

        Returns:
            torch.nn.Module: The layer module.

        Raises:
            ValueError: If no layer with the given name exists.
        """
        for name, module in self._model.named_modules():
            if name == layer_name:
                return module

        available_layers = [name for name, _ in self._model.named_modules() if name]
        raise ValueError(
            f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}"
            "..."
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Collect basic information about the model.

        Returns:
            Dict[str, Any]: Dictionary containing model metadata such as
            name, type, total parameters, device, and repository info.
        """
        total_params = sum(p.numel() for p in self._model.parameters())
        conv_layers = self.get_conv_layers()

        return {
            "name": self._model_name,
            "type": "pytorch_hub",
            "class": type(self._model).__name__,
            "total_parameters": total_params,
            "num_conv_layers": len(conv_layers),
            "device": str(next(self._model.parameters()).device),
            "sample_conv_layers": conv_layers[-3:] if conv_layers else [],
            "repo": self.repo,
            "pretrained": self.pretrained,
        }

    @property
    def pytorch_model(self) -> nn.Module:
        """
        Get the underlying PyTorch model instance.

        Returns:
            torch.nn.Module: The wrapped PyTorch _model.
        """
        return self._model

    def get_predictions(self, images: Tensor) -> Tensor:
        """
        Return val_indices for predicted classes.

        Args:
            images (Tensor): batch input images [B, C, H, W]

        Returns:
            Tensor: predicted val_indices [B]
        """
        self._model.eval()
        with torch.no_grad():
            return self._model(images)

    def _compute_model_hash(self, model: nn.Module) -> str:
        """
        Compute a hash of the model's state dict for verification.
        
        Args:
            model: PyTorch model to hash
            
        Returns:
            str: SHA256 hash of the model parameters
        """
        # Create hash of model parameters for verification
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:8]  # Short hash for logging
