from abc import ABC, abstractmethod

from torch import Tensor

from src.pipeline_moduls.xai_methods.base.dataclasses.explainer_result import (
    ExplainerResult,
)


class XAIInterface(ABC):
    @abstractmethod
    def explain(
        self, images: Tensor, target_labels: Tensor, top_k: int
    ) -> ExplainerResult:
        """
        Generates explanations and evaluates model predictions.

        Args:
            images (torch.Tensor): Input batch of images with shape [B, C, H, W].
            target_labels (torch.Tensor): Ground truth labels for each image in the
             batch, shape [B].
            top_k (int): Number of top class predictions to extract per image
             (e.g., top-1, top-5).

        Returns:
            ExplainerResult: Object containing attributions, predictions, confidences,
             and evaluation metrics.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return the name identifier of this explainer"""
