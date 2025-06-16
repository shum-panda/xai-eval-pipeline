from abc import ABC, abstractmethod

import torch
from torch import nn

from utilis.explainer_result import ExplainerResult


class BaseExplainer(ABC):
    """Abstract base class for all XAI explainers - simplified without BatchProcessor"""

    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        self.config = kwargs

    def explain(self, images: torch.Tensor, target_labels: torch.Tensor) -> ExplainerResult:
        """
        Template method - generates explanations and evaluates predictions

        Args:
            images: Input images tensor [B, C, H, W]
            target_labels: Ground truth labels tensor [B]

        Returns:
            ExplainerResult with attributions and evaluation
        """
        # Generate attributions
        attributions = self._compute_attributions(images)

        # Get predictions
        predictions = self._get_predictions(images)

        return ExplainerResult(
            attributions=attributions,
            predictions=predictions,
            target_labels=target_labels
        )

    def _get_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Default prediction method - can be overridden by explainers
        that capture predictions during attribution computation
        """
        with torch.no_grad():
            return self.model(images)

    @abstractmethod
    def _compute_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute attributions for the input images

        Args:
            images: Input images tensor [B, C, H, W]

        Returns:
            Attribution tensor with same spatial dimensions as input [B, C, H, W] or [B, H, W]
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name identifier of this explainer"""
        pass