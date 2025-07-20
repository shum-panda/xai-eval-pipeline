import logging
from abc import abstractmethod

import torch
from torch import nn

from pipeline_moduls.xai_methods.base.config_validation_result import (
    ConfigValidationResult,
)
from pipeline_moduls.xai_methods.base.explainer_result import ExplainerResult
from pipeline_moduls.xai_methods.base.validation_result import ValidationResult
from pipeline_moduls.xai_methods.base.xai_interface import XAIInterface


class BaseExplainer(XAIInterface):
    """Abstract base class for all XAI explainers - simplified without BatchProcessor"""

    def __init__(self, model: nn.Module, use_defaults: bool, **kwargs):
        self._model = model
        self._logger = logging.getLogger(self.__class__.__name__)

        # Runtime-Validierung der Parameter
        self._use_defaults = use_defaults
        self.config_validation = self.check_input(**kwargs)

        # Log Validierung-Ergebnis
        self._log_validation_result()

        # Setup mit validierten Parametern
        self._setup_with_validated_params(**kwargs)

    def explain(
        self, images: torch.Tensor, target_labels: torch.Tensor
    ) -> ExplainerResult:
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
            target_labels=target_labels,
        )

    def _get_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Default prediction method - can be overridden by explainers
        that capture predictions during attribution computation
        """
        with torch.no_grad():
            return self._model(images)

    @abstractmethod
    def _compute_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute attributions for the input images

        Args:
            images: Input images tensor [B, C, H, W]

        Returns:
            Attribution tensor with same spatial dimensions as input [B, C, H, W] or
             [B, H, W]
        """

    def _log_validation_result(self):
        """Log das Validierungs-Ergebnis für User-Feedback"""
        result = self.config_validation

        if result.status == ValidationResult.VALID:
            self._logger.info("All parameters valid")

        elif result.status == ValidationResult.MISSING_USING_DEFAULTS:
            self._logger.warning("Using default values for missing parameters:")
            for param, default_val in result.defaults_used.items():
                self._logger.warning(f"   {param}: {default_val} (default)")
            self._logger.warning(
                "Set 'use_defaults: false' in config to make this an error"
            )

        elif result.status == ValidationResult.INVALID:
            self._logger.error("Invalid parameters detected:")
            for param in result.invalid_params:
                self._logger.error(f"   {param}")
            raise ValueError(f"Invalid configuration: {result.message}")

    def check_input(self, **kwargs) -> ConfigValidationResult:
        """
        Validate input parameters at runtime.
        Each XAI method implements its own validation logic.

        Args:
            **kwargs: Parameters from the configuration.

        Returns:
            ConfigValidationResult indicating the validation status.
        """
        pass

    @abstractmethod
    def _setup_with_validated_params(self, **kwargs):
        """Setup der Explainer-spezifischen Parameter nach Validierung"""
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the name identifier of this explainer"""
