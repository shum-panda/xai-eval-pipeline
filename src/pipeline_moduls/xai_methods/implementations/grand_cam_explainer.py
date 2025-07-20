from dataclasses import dataclass
from typing import Optional

import torch
from captum.attr import LayerGradCam

from control.utils.with_cuda_cleanup import with_cuda_cleanup
from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from pipeline_moduls.xai_methods.base.config_validation_result import (
    ConfigValidationResult,
)
from pipeline_moduls.xai_methods.base.explainer_result import ExplainerResult
from pipeline_moduls.xai_methods.base.validation_result import ValidationResult


@dataclass
class GradCAMConfig:
    """GradCAM-spezifische Konfiguration"""

    target_layer: Optional[str] = None  # None = auto-detect letzter conv layer
    relu_attributions: bool = True
    interpolate_mode: str = "bilinear"
    use_cuda: bool = True
    guided_backprop: bool = False

    @classmethod
    def get_defaults(cls) -> dict:
        """Hole Standard-Parameter"""
        return {
            "target_layer": None,
            "relu_attributions": True,
            "interpolate_mode": "bilinear",
            "use_cuda": True,
            "guided_backprop": False,
        }

    @classmethod
    def get_required_params(cls) -> list:
        """Hole erforderliche Parameter (die nicht None sein dürfen)"""
        return ["relu_attributions", "interpolate_mode", "use_cuda"]


class GradCamExplainer(BaseExplainer):
    @with_cuda_cleanup
    def _compute_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute GradCAM attributions using Captum's LayerGradCam.

        Args:
            images: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            GradCAM attributions tensor
        """
        try:
            # Get predictions and target classes
            with torch.no_grad():
                predictions = self._model(images)
                target_classes = predictions.argmax(dim=1)

            # Compute GradCAM attributions
            attributions = self.gradcam.attribute(
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
                self.gradcam, "relu_attributions"
            ):
                attributions = torch.relu(attributions)

            self._logger.debug(
                f"Computed GradCAM attributions with shape: {attributions.shape}"
            )
            return attributions

        except Exception as e:
            self._logger.error(
                f"Error computing GradCAM attributions: {str(e)}"
            )  # todo add correct Exception Handling
            # Fallback to dummy attributions for robustness
            self._logger.warning("Falling back to dummy attributions")
            attributions = torch.randn_like(images)
            if self.relu_attributions:
                attributions = torch.relu(attributions)
            return attributions

    def _select_target_layer(self, model, layer_idx):
        """
        Select target layer for GradCAM.

        Args:
            model: PyTorch _model_name
            layer_idx: Layer index (-1 for last layer)

        Returns:
            Selected layer module
        """
        if isinstance(layer_idx, int):
            # Get all modules that could be target layers (Conv2d)
            conv_modules = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_modules.append((name, module))

            if not conv_modules:
                self._logger.warning("No Conv2d layers found, using _model_name itself")
                return model

            if layer_idx == -1:
                # Use last conv layer
                selected = conv_modules[-1]
                self._logger.info(f"Selected last conv layer: {selected[0]}")
                return selected[1]
            elif 0 <= layer_idx < len(conv_modules):
                selected = conv_modules[layer_idx]
                self._logger.info(f"Selected conv layer {layer_idx}: {selected[0]}")
                return selected[1]
            else:
                self._logger.warning(
                    f"Layer index {layer_idx} out of range, using last layer"
                )
                return conv_modules[-1][1]
        else:
            # Assume it's already a module
            return layer_idx

    @with_cuda_cleanup
    def explain_with_target_class(
        self,
        images: torch.Tensor,
        target_classes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> ExplainerResult:
        """
        Explain with specific target classes instead of predicted classes.

        Args:
            images: Input images
            target_classes: Specific classes to explain
            target_labels: Ground truth labels for evaluation

        Returns:
            ExplainerResult with attributions and evaluation
        """
        try:
            # Get predictions for evaluation (but use target_classes for explanation)
            predictions = self._get_predictions(images)

            # Compute GradCAM with specific target classes
            attributions = self.gradcam.attribute(
                inputs=images,
                target=target_classes,
                relu_attributions=self.relu_attributions,
            )

            # Interpolate to input size
            if attributions.shape[-2:] != images.shape[-2:]:
                attributions = torch.nn.functional.interpolate(
                    attributions,
                    size=images.shape[-2:],
                    mode=self.interpolate_mode,
                    align_corners=False,
                )

            if self.relu_attributions and not hasattr(
                self.gradcam, "relu_attributions"
            ):
                attributions = torch.relu(attributions)

            return ExplainerResult(
                attributions=attributions,
                predictions=predictions,
                target_labels=target_labels,
            )

        except Exception as e:
            self._logger.error(f"Error in targeted GradCAM: {str(e)}")
            # Fallback to standard explanation
            self._logger.warning("Falling back to standard GradCAM explanation")
            return self.explain(images, target_labels)

    def check_input(self, **kwargs) -> ConfigValidationResult:
        """
        Runtime-Validierung für GradCAM Parameter
        """
        defaults = GradCAMConfig.get_defaults()
        required = GradCAMConfig.get_required_params()

        missing_params = []
        invalid_params = []
        defaults_used = {}

        # 1. Prüfe fehlende Parameter
        for param in required:
            if param not in kwargs:
                missing_params.append(param)
                if self._use_defaults and param in defaults:
                    defaults_used[param] = defaults[param]
                    kwargs[param] = defaults[param]  # Setze default

        # 2. Prüfe target_layer (spezielle Logik)
        if "target_layer" not in kwargs or kwargs["target_layer"] is None:
            if self._use_defaults:
                # Auto-detect letzter conv layer
                conv_layers = (
                    self._model.get_conv_layers()
                    if hasattr(self._model, "get_conv_layers")
                    else []
                )
                if conv_layers:
                    auto_layer = conv_layers[-1]
                    defaults_used["target_layer"] = f"{auto_layer} (auto-detected)"
                    kwargs["target_layer"] = auto_layer
                else:
                    missing_params.append("target_layer")

        # 3. Validiere Parameter-Typen
        if "n_steps" in kwargs and not isinstance(kwargs.get("n_steps"), int):
            invalid_params.append("n_steps (must be int)")

        if "relu_attributions" in kwargs and not isinstance(
            kwargs.get("relu_attributions"), bool
        ):
            invalid_params.append("relu_attributions (must be bool)")

        # 4. Bestimme Validierungs-Status
        if invalid_params:
            return ConfigValidationResult(
                status=ValidationResult.INVALID,
                message=f"Invalid parameters: {invalid_params}",
                invalid_params=invalid_params,
            )
        elif missing_params and not self._use_defaults:
            return ConfigValidationResult(
                status=ValidationResult.INVALID,
                message=f"Missing required parameters: {missing_params}",
                missing_params=missing_params,
            )
        elif defaults_used:
            return ConfigValidationResult(
                status=ValidationResult.MISSING_USING_DEFAULTS,
                message=f"Using defaults for: {list(defaults_used.keys())}",
                missing_params=missing_params,
                defaults_used=defaults_used,
            )
        else:
            return ConfigValidationResult(
                status=ValidationResult.VALID, message="All parameters valid"
            )

    def _setup_with_validated_params(self, **kwargs):
        """Setup GradCAM mit validierten Parametern"""
        # Jetzt sind alle Parameter validiert und können sicher verwendet werden
        self.target_layer = kwargs.get("target_layer")
        self.relu_attributions = kwargs.get("relu_attributions", True)
        self.interpolate_mode = kwargs.get("interpolate_mode", "bilinear")
        self.use_cuda = kwargs.get("use_cuda", True)
        self.guided_backprop = kwargs.get("guided_backprop", False)

        # Setup GradCAM mit Captum

        if isinstance(self.target_layer, str):
            layer_obj = self._model.get_layer_by_name(self.target_layer)
        else:
            layer_obj = self.target_layer

        self.gradcam = LayerGradCam(self._model, layer_obj)
        self._model.eval()

        self._logger.info(
            f"GradCAM setup complete with target_layer: {self.target_layer}"
        )

    @classmethod
    def get_name(cls) -> str:
        """Return explainer name"""
        return "gradcam"
