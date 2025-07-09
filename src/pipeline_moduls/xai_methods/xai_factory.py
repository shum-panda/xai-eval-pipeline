from typing import List

from torch import nn

from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from pipeline_moduls.xai_methods.explainer_registry import ExplainerRegistry
from pipeline_moduls.xai_methods.implementations.grand_cam_explainer import (
    GradCamExplainer,
)
from pipeline_moduls.xai_methods.implementations.integraded_gradients import (
    IntegratedGradientsExplainer,
)
from pipeline_moduls.xai_methods.implementations.occlusion_explainer import (
    OcclusionExplainer,
)


class XAIFactory:
    """Factory for creating XAI explainers using the registry"""

    def __init__(self):
        self.registry = ExplainerRegistry.get_instance()
        self._register_default_explainers()

    def _register_default_explainers(self):
        """Register default explainer implementations"""
        self.registry.register("gradcam", GradCamExplainer)
        self.registry.register("occlusion", OcclusionExplainer)
        self.registry.register("integrated_gradients", IntegratedGradientsExplainer)

    def create_explainer(self, name: str, model: nn.Module, **kwargs) -> BaseExplainer:
        """
        Create an explainer instance

        Args:
            name: Name of the explainer to create
            model: PyTorch model_name to explain
            **kwargs: Additional arguments for explainer initialization

        Returns:
            Configured explainer instance
        """
        explainer_class = self.registry.get(name)
        return explainer_class(model, **kwargs)

    def list_available_explainers(self) -> List[str]:
        """List all available explainer types"""
        return self.registry.list_available()
