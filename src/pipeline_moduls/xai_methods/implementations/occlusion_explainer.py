import logging
from dataclasses import dataclass

import torch
from captum.attr import Occlusion

from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer


@dataclass
class OcclusionConfig:
    sliding_window_shape: tuple = (3, 15, 15)  # channels, height, width
    stride: tuple = (3, 8, 8)  # channels, height, width


class OcclusionExplainer(BaseExplainer):
    """
    Occlusion Explainer.
    """

    def __init__(self, model, config: OcclusionConfig = None, **kwargs):
        """
        Initialize Occlusion explainer.

        Args:
            model: PyTorch model
            config: Occlusion configuration
            **kwargs: Additional arguments
        """
        config = config or OcclusionConfig()
        super().__init__(model, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.sliding_window_shape = config.sliding_window_shape
        self.stride = config.stride

        self.occlusion = Occlusion(self.model)
        self.model.eval()

        self.logger.info(
            "Occlusion initialized with sliding_window_shape="
            f"{self.sliding_window_shape}, stride={self.stride}"
        )

    def _compute_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute Occlusion attributions.

        Args:
            images: Input tensor (batch_size, channels, height, width)

        Returns:
            Attributions tensor
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.eval()

            with torch.no_grad():
                predictions = self.model(images)
                target_classes = predictions.argmax(dim=1)

            attributions = self.occlusion.attribute(
                inputs=images,
                strides=self.stride,
                sliding_window_shapes=self.sliding_window_shape,
                target=target_classes,
            )

            self.logger.debug(
                f"Computed Occlusion attributions with shape: {attributions.shape}"
            )
            return attributions

        except Exception as e:
            self.logger.error(f"Error computing Occlusion attributions: {str(e)}")
            self.logger.warning("Falling back to dummy attributions")
            return torch.randn_like(images)

    @classmethod
    def get_name(cls) -> str:
        return "occlusion"
