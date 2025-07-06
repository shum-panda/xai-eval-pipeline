import logging
from dataclasses import dataclass

import torch
from captum.attr import LayerGradCam

from control.dataclasses.explainer_result import ExplainerResult
from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer


@dataclass
class GradCAMConfig:
    target_layer: int = -1  # -1 for last conv layer
    relu_attributions: bool = True
    interpolate_mode: str = 'bilinear'


class GradCamExplainer(BaseExplainer):
    """
    GradCAM Explainer - simplified without BatchProcessor.

    Processes images directly and handles predictions efficiently.
    """

    def __init__(self, model, config: GradCAMConfig = None, **kwargs):
        """
        Initialize GradCAM explainer

        Args:
            model: PyTorch model_name
            config: GradCAM configuration
            **kwargs: Additional arguments
        """
        config = config or GradCAMConfig()
        super().__init__(model, **kwargs)

        # Logger
        self.logger = logging.getLogger(__name__)

        # GradCAM specific configuration
        self.layer = config.target_layer
        self.relu_attributions = config.relu_attributions
        self.interpolate_mode = config.interpolate_mode

        # Setup target layer and GradCAM
        self.target_layer = self._select_target_layer(model, config.target_layer)
        self.gradcam = LayerGradCam(model, self.target_layer)

        # Ensure model_name is in evaluation mode
        self.model.eval()

        self.logger.info(f"GradCAM initialized with target layer: {self.target_layer}")

    def _compute_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute GradCAM attributions using Captum's LayerGradCam.

        Args:
            images: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            GradCAM attributions tensor
        """
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Ensure model_name is in eval mode
            self.model.eval()

            # Get predictions and target classes
            with torch.no_grad():
                predictions = self.model(images)
                target_classes = predictions.argmax(dim=1)

            # Compute GradCAM attributions
            attributions = self.gradcam.attribute(
                inputs=images,
                target=target_classes,
                relu_attributions=self.relu_attributions
            )

            # Interpolate attributions to match input image size
            if attributions.shape[-2:] != images.shape[-2:]:
                attributions = torch.nn.functional.interpolate(
                    attributions,
                    size=images.shape[-2:],
                    mode=self.interpolate_mode,
                    align_corners=False
                )

            # Ensure attributions are non-negative if relu_attributions is True
            if self.relu_attributions and not hasattr(self.gradcam, 'relu_attributions'):
                attributions = torch.relu(attributions)

            self.logger.debug(f"Computed GradCAM attributions with shape: {attributions.shape}")
            return attributions

        except Exception as e:
            self.logger.error(f"Error computing GradCAM attributions: {str(e)}")#todo add correct Exception Handling
            # Fallback to dummy attributions for robustness
            self.logger.warning("Falling back to dummy attributions")
            attributions = torch.randn_like(images)
            if self.relu_attributions:
                attributions = torch.relu(attributions)
            return attributions

    def _select_target_layer(self, model, layer_idx):
        """
        Select target layer for GradCAM.

        Args:
            model: PyTorch model_name
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
                self.logger.warning("No Conv2d layers found, using model_name itself")
                return model

            if layer_idx == -1:
                # Use last conv layer
                selected = conv_modules[-1]
                self.logger.info(f"Selected last conv layer: {selected[0]}")
                return selected[1]
            elif 0 <= layer_idx < len(conv_modules):
                selected = conv_modules[layer_idx]
                self.logger.info(f"Selected conv layer {layer_idx}: {selected[0]}")
                return selected[1]
            else:
                self.logger.warning(f"Layer index {layer_idx} out of range, using last layer")
                return conv_modules[-1][1]
        else:
            # Assume it's already a module
            return layer_idx

    def explain_with_target_class(self, images: torch.Tensor, target_classes: torch.Tensor,
                                  target_labels: torch.Tensor) -> ExplainerResult:
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
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model.eval()

            # Get predictions for evaluation (but use target_classes for explanation)
            predictions = self._get_predictions(images)

            # Compute GradCAM with specific target classes
            attributions = self.gradcam.attribute(
                inputs=images,
                target=target_classes,
                relu_attributions=self.relu_attributions
            )

            # Interpolate to input size
            if attributions.shape[-2:] != images.shape[-2:]:
                attributions = torch.nn.functional.interpolate(
                    attributions,
                    size=images.shape[-2:],
                    mode=self.interpolate_mode,
                    align_corners=False
                )

            if self.relu_attributions and not hasattr(self.gradcam, 'relu_attributions'):
                attributions = torch.relu(attributions)

            return ExplainerResult(
                attributions=attributions,
                predictions=predictions,
                target_labels=target_labels
            )

        except Exception as e:
            self.logger.error(f"Error in targeted GradCAM: {str(e)}")
            # Fallback to standard explanation
            self.logger.warning("Falling back to standard GradCAM explanation")
            return self.explain(images, target_labels)

    @classmethod
    def get_name(cls)-> str:
        """Return explainer name"""
        return "gradcam"

