from abc import ABC, abstractmethod

import torch

from control.utils.dataclasses.explainer_result import ExplainerResult


class XAIInterface(ABC):
    @abstractmethod
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

    @abstractmethod
    def get_name(self) -> str:
        """Return the name identifier of this explainer"""
