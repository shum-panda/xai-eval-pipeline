import logging
from abc import abstractmethod

import torch
from functorch.dim import Tensor
from torch import nn

from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from src.pipeline_moduls.xai_methods.base.dataclasses.explainer_result import (
    ExplainerResult,
)
from src.pipeline_moduls.xai_methods.base.xai_interface import XAIInterface


class BaseExplainer(XAIInterface):
    """Abstract base class for all XAI explainers"""

    def __init__(self, model: nn.Module, use_defaults: bool, **kwargs):
        self._model = model
        self._logger = logging.getLogger(self.__class__.__name__)

        # Runtime-Validierung der Parameter
        self._use_defaults = use_defaults
        self.validated_config = self.check_input(**kwargs)

        # Setup mit validierten Parametern
        self._setup_with_validated_params(self.validated_config)

    def explain(
        self, images: torch.Tensor, target_labels: torch.Tensor, top_k: int
    ) -> ExplainerResult:
        """
        Template method - generates explanations and evaluates predictions

        Args:
            images: Input images tensor [B, C, H, W]
            target_labels: Ground truth labels tensor [B]
            top_k: length of predictions

        Returns:
            ExplainerResult with attributions and evaluation
        """
        # Get predictions and target classes
        logits = self._get_predictions(images)
        probs = torch.softmax(logits, dim=1)
        confidence, predictions = torch.max(probs, dim=1)
        target_classes = predictions
        topk_confidences, topk_predictions = torch.topk(probs, k=top_k, dim=1)

        # Generate attributions
        attributions = self._compute_attributions(images, target_classes)
        return ExplainerResult(
            attributions=attributions,
            probabilities=probs,
            predictions=predictions,
            confidence=confidence,
            target_labels=target_labels,
            topk_predictions=topk_predictions,
            topk_confidences=topk_confidences,
        )

    def _get_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model to obtain predictions for the given
        input images.

        This method can be overridden by explainers that compute or cache predictions
        internally during the attribution process.

        Args:
            images (torch.Tensor): Input images of shape (N, C, H, W), as expected by
            the model.

        Returns:
            torch.Tensor: Model outputs (e.g., logits or probabilities), depending on
            the model architecture.
        """
        with torch.no_grad():
            return self._model(images)

    @abstractmethod
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        """
        Compute attributions for the input images

        Args:
            images: Input images tensor [B, C, H, W]

        Returns:
            Attribution tensor with same spatial dimensions as input [B, C, H, W] or
             [B, H, W]
        """

    @abstractmethod
    def check_input(self, **kwargs) -> BaseXAIConfig:
        """
        Validate input parameters at runtime.
        Each XAI method implements its own validation logic.

        Args:
            **kwargs: Parameters from the configuration.

        Returns:
            Base
        """
        pass

    @abstractmethod
    def _setup_with_validated_params(self, config: BaseXAIConfig):
        """Setup der Explainer-spezifischen Parameter nach Validierung"""
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the name identifier of this explainer"""
