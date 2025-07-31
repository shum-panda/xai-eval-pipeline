from typing import Any, Dict

import torch
from torch import Tensor
from captum.attr import GuidedBackprop  # type: ignore

from src.pipeline_moduls.models.base.interface.xai_model import XAIModel
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from src.pipeline_moduls.xai_methods.impl.guided_backprop.guided_backprop_config import GuidedBackpropConfig
from utils.with_cuda_cleanup import with_cuda_cleanup


class GuidedBackpropExplainer(BaseExplainer):
    """
    Explainer for Guided Backpropagation using Captum.
    """

    @property
    def parameters(self) -> Dict[str, str]:
        return {}

    def __init__(self, model: XAIModel, use_defaults: bool = True, **kwargs: object) -> None:
        self.guided_backprop = None
        super().__init__(model, use_defaults, **kwargs)

    @with_cuda_cleanup
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        """
        Compute guided backpropagation attributions.

        Args:
            images (Tensor): Input images of shape (B, C, H, W).
            target_classes (Tensor): Class indices to compute gradients for.

        Returns:
            Tensor: Attributions of shape (B, H, W) after averaging across channels.
        """
        if self.guided_backprop is None:
            raise RuntimeError("GuidedBackprop was not initialized.")

        attributions = self.guided_backprop.attribute(images, target=target_classes)

        # Optional: Average over channels to get 2D heatmaps
        if attributions.dim() == 4 and attributions.shape[1] == 3:
            attributions = attributions.mean(dim=1)

        attributions = torch.abs(attributions)
        return attributions.detach().cpu()

    def check_input(self, **kwargs: Any) -> BaseXAIConfig:
        config = GuidedBackpropConfig(**kwargs)
        config.validate()
        return config

    def _setup_with_validated_params(self, config: BaseXAIConfig) -> None:
        if not isinstance(config, GuidedBackpropConfig):
            raise TypeError("Expected GuidedBackpropConfig.")
        self.guided_backprop = GuidedBackprop(self._model.pytorch_model)

    @classmethod
    def get_name(cls) -> str:
        return "guided_backprop"
