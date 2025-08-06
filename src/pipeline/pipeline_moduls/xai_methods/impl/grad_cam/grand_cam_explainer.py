from typing import Any, Dict, Union

import torch
from captum.attr import LayerGradCam  # type: ignore
from torch import Tensor, nn

from pipeline.utils import with_cuda_cleanup
from src.pipeline.pipeline_moduls.models.base.xai_model import XAIModel
from src.pipeline.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from src.pipeline.pipeline_moduls.xai_methods.impl.grad_cam.grad_cam_config import (
    GradCAMConfig,
)


class GradCamExplainer(BaseExplainer):
    """
    Explainer class implementing GradCAM using Captum's LayerGradCam.
    """

    def __init__(
        self, model: XAIModel, use_defaults: bool = True, **kwargs: object
    ) -> None:
        """
        Initialize GradCamExplainer.

        Args:
            model (nn.Module): Model to explain.
            use_defaults (bool): Whether to use default configuration.
            **kwargs: Additional keyword arguments.
        """
        self.grad_cam = None
        self.target_layer = None
        self.relu_attributions = True
        self.interpolate_mode = "bilinear"

        super().__init__(model, use_defaults, **kwargs)
        self._logger.debug(f"GradCam after init: {self.grad_cam}")

    @with_cuda_cleanup
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        """
        Compute GradCAM attributions using Captum's LayerGradCam.

        Args:
            images (Tensor): Input tensor of shape (B, C, H, W).
            target_classes (Tensor): Target class indices for attribution.

        Returns:
            Tensor: GradCAM attribution maps with same spatial size as `images`.
        """
        if self.grad_cam is None:
            raise RuntimeError("GradCAM has not been initialized.")

        try:
            attributions: Tensor = self.grad_cam.attribute(
                inputs=images,
                target=target_classes,
                relu_attributions=self.relu_attributions,
            )

            # Resize to input image size if necessary
            if attributions.shape[-2:] != images.shape[-2:]:
                attributions = torch.nn.functional.interpolate(
                    attributions,
                    size=images.shape[-2:],
                    mode=self.interpolate_mode,
                    align_corners=False,
                )

            # Ensure consistent output shape (B, H, W)
            attributions = attributions.max(dim=1)[0]

            self._logger.debug(
                f"Computed GradCAM attributions with shape: {attributions.shape}"
            )
            return attributions.detach().cpu()

        except Exception as e:
            self._logger.error(f"Error computing GradCAM attributions: {str(e)}")
            raise

    def _check_input(self, **kwargs: Any) -> BaseXAIConfig:
        """
        Validates and returns a GradCAMConfig based on input kwargs.

        Args:
            **kwargs: Raw configuration values.

        Returns:
            GradCAMConfig: Validated configuration object.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if self._use_defaults:
            # Default-Werte hart hinzufügen, falls nicht gesetzt
            defaults = GradCAMConfig.get_defaults()
            for k, v in defaults.items():
                kwargs.setdefault(k, v)
        try:
            self._logger.info(f"kwargs {kwargs}")
            config = GradCAMConfig(use_defaults=self._use_defaults, **kwargs)
            config.validate()
        except (TypeError, ValueError) as e:
            self._logger.error(f"Invalid config: {e}")
            raise

        try:
            config.target_layer = self._select_target_layer(config.target_layer)
        except Exception as e:
            raise ValueError(f"Invalid target_layer: {e}")

        return config

    def _select_target_layer(self, layer_idx: Union[str, int]) -> str:
        """
        Select a convolutional layer as the GradCAM target.

        Args:
            layer_idx (Union[str, int]): Index or name of the desired layer.

        Returns:
            str: name of The selected target layer.
        """
        model = self._model.pytorch_model
        conv_modules: list[tuple[str, nn.Module]] = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_modules.append((name, module))

        if not conv_modules:
            self._logger.warning("No Conv2d layers found. Returning the _model itself.")
            raise ValueError("this GradCam expected Convolutional Layer ")

        if isinstance(layer_idx, str):
            for name, module in conv_modules:
                if name == layer_idx:
                    selected = (name, module)
                    break
            else:
                self._logger.warning(f"Layer name {layer_idx} not found.")
                raise ValueError(f"Layer name {layer_idx} not found.")
        elif isinstance(layer_idx, int):
            if layer_idx == -1 or layer_idx >= len(conv_modules):
                self._logger.info("Using last convolutional layer.")
                selected = conv_modules[-1]
            elif 0 <= layer_idx < len(conv_modules):
                selected = conv_modules[layer_idx]
            else:
                self._logger.warning(
                    f"Layer index {layer_idx} out of range. Using last layer."
                )
                selected = conv_modules[-1]
        else:
            raise TypeError("layer_idx must be a string or integer.")

        return selected[0]

    def _setup_with_validated_params(self, config: BaseXAIConfig) -> None:
        """
        Sets up the explainer with a validated GradCAMConfig.

        Args:
            config (BaseXAIConfig): Validated GradCAM configuration.

        Raises:
            TypeError: If the config is not a GradCAMConfig.
        """
        if not isinstance(config, GradCAMConfig):
            raise TypeError(f"Expected GradCAMConfig, got {type(config).__name__}")

        self.target_layer = config.target_layer
        self.relu_attributions = config.relu_attributions
        self.interpolate_mode = config.interpolate_mode

        layer_obj = (
            self._model.get_layer_by_name(self.target_layer)
            if isinstance(self.target_layer, str)
            else self.target_layer
        )

        self.grad_cam = LayerGradCam(self._model.pytorch_model, layer_obj)

        self._logger.info(
            f"GradCAM setup complete with target_layer: {self.target_layer}, "
            f"interpolate_mode: {self.interpolate_mode}, "
            f"relu_attributions: {self.relu_attributions}"
            f"grad_cam: {self.grad_cam}"
        )

    @property
    def parameters(self) -> Dict[str, str]:
        """Returns a dictionary of parameter names and their stringified values."""
        return {
            "target_layer": str(self.target_layer),
            "relu_attributions": str(self.relu_attributions),
            "interpolate_mode": str(self.interpolate_mode),
        }

    @classmethod
    def get_name(cls) -> str:
        """
        Returns:
            str: Unique name of the explainer.
        """
        return "grad_cam"
