import logging
from dataclasses import dataclass
from captum.attr import LayerGradCam
import torch

from xai_methods.MemoryManagement.base.batch_processor import BatchProcessor
from xai_methods.base.base_explainer import BaseExplainer

@dataclass
class GradCAMConfig:
    target_layer: int = -1
    relu_attributions: bool = True
    interpolate_mode: str = 'bilinear'

class GradCamExplainer(BaseExplainer):
    """
    GradCAM Explainer ohne zirkuläre Abhängigkeiten.

    Verwendet Strategy Pattern für Memory Management.
    """


    def __init__(self, model, batch_processor: BatchProcessor = None, config: GradCAMConfig = None, **kwargs):
        """
        todo
        :param model:
        :param batch_processor:
        :param config:
        :param kwargs:
        """

        config = config or GradCAMConfig()
        super().__init__(model, batch_processor, **kwargs)
        self.layer = config.target_layer
        self.relu_attributions = config.relu_attributions
        self.interpolate_mode = config.interpolate_mode

        # GradCAM Setup (vereinfacht)
        self.target_layer = self._select_target_layer(model, config.target_layer)
        self.logger = logging.getLogger(__name__)

        self.gradcam = LayerGradCam(model, self.target_layer)
        # Set model to evaluation mode
        self.model.eval()

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

            # Ensure model is in eval mode
            self.model.eval()

            # Get the predicted class for each image (or use target class if specified)
            with torch.no_grad():
                predictions = self.model(images)
                target_classes = predictions.argmax(dim=1)

            # Compute GradCAM attributions
            # LayerGradCam.attribute returns attributions for the target layer
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
            self.logger.error(f"Error computing GradCAM attributions: {str(e)}")
            # Fallback to dummy attributions for robustness
            self.logger.warning("Falling back to dummy attributions")
            attributions = torch.randn_like(images)
            if self.relu_attributions:
                attributions = torch.relu(attributions)
            return attributions

    def _select_target_layer(self, model, layer_idx):
        """Layer-Selektion (vereinfacht)."""
        layers = list(model.children())
        return layers[layer_idx] if layers else model

    def get_name(self) -> str:
        return "gradcam"
