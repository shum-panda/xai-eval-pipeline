import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from src.pipeline_moduls.xai_methods.base.dataclasses.explainer_result import (
    ExplainerResult,
)
from src.pipeline_moduls.xai_methods.base.xai_interface import XAIInterface


class BaseExplainer(XAIInterface, ABC):
    """
    Abstract base class for all XAI explainers.

    Defines the structure and lifecycle of explainers including:
    - input validation
    - setup with validated parameters
    - explanation generation
    """

    def __init__(self, model: nn.Module, use_defaults: bool, **kwargs: Any) -> None:
        """
        Initialize the explainer.

        Args:
            model (nn.Module): The PyTorch model to be explained.
            use_defaults (bool): Whether to use default configuration values.
            **kwargs (Any): Additional arguments for the explainer configuration.
        """
        self._model: nn.Module = model
        self._logger = logging.getLogger(self.__class__.__name__)
        self._use_defaults = use_defaults

        # Validate and setup configuration
        self.validated_config: BaseXAIConfig = self.check_input(**kwargs)
        self._setup_with_validated_params(self.validated_config)

    def explain(
        self,
        images: Tensor,
        target_labels: Tensor,
        top_k: int,
    ) -> ExplainerResult:
        """
        Main template method to generate explanations and record prediction outputs.

        Args:
            images (Tensor): Input images tensor of shape (B, C, H, W).
            target_labels (Tensor): Ground-truth labels tensor of shape (B,).
            top_k (int): Number of top predictions to return.

        Returns:
            ExplainerResult: Object containing attributions and prediction metadata.
        """
        logits = self._get_predictions(images)
        probs = torch.softmax(logits, dim=1)
        confidence, predictions = torch.max(probs, dim=1)
        target_classes = predictions
        topk_confidences, topk_predictions = torch.topk(probs, k=top_k, dim=1)

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

    def _get_predictions(self, images: Tensor) -> Tensor:
        """
        Compute model outputs from input images.

        Can be overridden by subclasses if predictions are computed differently.

        Args:
            images (Tensor): Input batch of images.

        Returns:
            Tensor: Model output tensor (e.g., logits).
        """
        self._model.eval()
        with torch.no_grad():
            return self._model(images)

    @abstractmethod
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        """
        Compute attributions (i.e., explanations) for given inputs.

        Args:
            images (Tensor): Input image tensor of shape (B, C, H, W).
            target_classes (Tensor): Target classes for attribution.

        Returns:
            Tensor: Attribution maps, typically of shape (B, H, W) or (B, C, H, W).
        """
        pass

    @abstractmethod
    def check_input(self, **kwargs: Any) -> BaseXAIConfig:
        """
        Validate and return a configuration object for the explainer.

        Args:
            **kwargs (Any): Raw configuration values from the user or config system.

        Returns:
            BaseXAIConfig: Validated configuration object.
        """
        pass

    @abstractmethod
    def _setup_with_validated_params(self, config: BaseXAIConfig) -> None:
        """
        Initialize internal parameters and components from the validated config.

        Args:
            config (BaseXAIConfig): Validated configuration for the explainer.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Return a unique name for this explainer, used for registration.

        Returns:
            str: The name identifier of the explainer.
        """
        pass
