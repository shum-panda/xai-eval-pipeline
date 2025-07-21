from typing import List

from torch import nn

from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from pipeline_moduls.xai_methods.explainer_registry import ExplainerRegistry
from pipeline_moduls.xai_methods.implementations.grad_cam.grand_cam_explainer import (
    GradCamExplainer,
)


class XAIFactory:
    """Factory for creating XAI explainers using the registry"""

    def __init__(self):
        self.registry = ExplainerRegistry.get_instance()
        self._register_default_explainers()

    def _register_default_explainers(self):
        """Register default explainer implementations"""
        self.registry.register("grad_cam", GradCamExplainer)

    def create_explainer(
        self, name: str, model: nn.Module, use_defaults: bool, **kwargs
    ) -> BaseExplainer:
        """
        Create and return an explainer instance.

        Args:
            name: The name of the explainer to create (must be registered).
            model: The PyTorch model to be explained.
            use_defaults: Whether to use default parameters for the explainer.
            **kwargs: Additional keyword arguments passed to the explainer constructor.

        Returns:
            A configured instance of the selected explainer.
        """
        explainer_class = self.registry.get(name)
        return explainer_class(model, use_defaults, **kwargs)

    def list_available_explainers(self) -> List[str]:
        """List all available explainer types"""
        return self.registry.list_available()
