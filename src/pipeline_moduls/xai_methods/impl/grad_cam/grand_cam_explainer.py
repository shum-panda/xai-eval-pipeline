from typing import Union

import torch
from captum.attr import LayerGradCam
from torch import Tensor

from src.control.utils.with_cuda_cleanup import with_cuda_cleanup
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from src.pipeline_moduls.xai_methods.impl.grad_cam.grad_cam_config import (
    GradCAMConfig,
)


class GradCamExplainer(BaseExplainer):

    def __init__(self, model, use_defaults=True, **kwargs):
        """Initialize GradCam Explainer."""
        super().__init__(model, use_defaults, **kwargs)
        self._logger.debug(f"GradCam after init: {self.grad_cam}")

    @with_cuda_cleanup
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        """
        Compute GradCAM attributions using Captum's LayerGradCam.

        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            target_classes: Input tensor of the target_classes

        Returns:
            GradCAM attributions tensor
        """
        try:
            # Compute GradCAM attributions
            attributions = self.grad_cam.attribute(
                inputs=images,
                target=target_classes,
                relu_attributions=self.relu_attributions,
            )

            # Interpolate attributions to match input image size
            if attributions.shape[-2:] != images.shape[-2:]:
                attributions = torch.nn.functional.interpolate(
                    attributions,
                    size=images.shape[-2:],
                    mode=self.interpolate_mode,
                    align_corners=False,
                )

            # Ensure attributions are non-negative if relu_attributions is True
            if self.relu_attributions and not hasattr(
                self.grad_cam, "relu_attributions"
            ):
                attributions = torch.relu(attributions)

            self._logger.debug(
                f"Computed GradCAM attributions with shape: {attributions.shape}"
            )
            return attributions

        except Exception as e:
            self._logger.error(f"Error computing GradCAM attributions: {str(e)}")
            raise e

    def _select_target_layer(self, layer_idx: Union[str, int]):
        """
        Select target layer for GradCAM.

        Args:
            layer_idx: Layer index (-1 for last layer)

        Returns:
            Selected layer module
        """
        model = self._model
        selected = layer_idx
        conv_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_modules.append((name, module))

        if not conv_modules:
            self._logger.warning("No Conv2d layers found, using _model_name itself")
            return model

        if not layer_idx or layer_idx == -1:
            selected = conv_modules[-1]
            self._logger.info(f"Selected last conv layer: {selected[0]}")
        elif isinstance(layer_idx, int):
            # Get all modules that could be target layers (Conv2d)
            if layer_idx == -1:
                # Use last conv layer
                selected = conv_modules[-1]
                self._logger.info(f"Selected last conv layer: {selected[0]}")
            elif 0 <= layer_idx < len(conv_modules):
                selected = conv_modules[layer_idx]
                self._logger.info(f"Selected conv layer {layer_idx}: {selected[0]}")
            else:
                self._logger.warning(
                    f"Layer index {layer_idx} out of range, using last layer"
                )
                selected = conv_modules[-1]
        return selected[1]

    def check_input(self, **kwargs) -> BaseXAIConfig:
        try:
            config = GradCAMConfig(**kwargs)  # wirft TypeError bei falschem Typ
            config.validate()  # wirft InvalidValueError bei falschem Wert
        except (TypeError, ValueError) as e:
            self._logger.error(f"Invalid config: {e}")
            raise

        try:
            config.target_layer = self._select_target_layer(config.target_layer)
        except Exception as e:
            raise ValueError(f"Invalid target_layer: {e}")

        return config

    def _setup_with_validated_params(self, config: BaseXAIConfig):
        """Setup GradCAM mit validierten Parametern"""
        if not isinstance(config, GradCAMConfig):
            raise TypeError(f"Expected GradCAMConfig, got {type(config).__name__}")

        self.target_layer = config.target_layer
        self.relu_attributions = config.relu_attributions
        self.interpolate_mode = config.interpolate_mode

        # Setup GradCAM mit Captum

        if isinstance(self.target_layer, str):
            layer_obj = self._model.get_layer_by_name(self.target_layer)
        else:
            layer_obj = self.target_layer

        self.grad_cam = LayerGradCam(self._model, layer_obj)
        self._model.eval()

        self._logger.info(
            f"GradCAM setup complete with target_layer: {self.target_layer}"
            f"interpolate_mode: {self.interpolate_mode},"
            f"relu_attributions: {self.relu_attributions}"
        )

    @classmethod
    def get_name(cls) -> str:
        """Return explainer name."""
        return "grad_cam"
