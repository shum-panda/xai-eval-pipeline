from typing import List

from torch import nn

from xai_methods.MemoryManagement.base.batch_processor import BatchProcessor
from xai_methods.MemoryManagement.implementations.direct_batch_processor import DirectBatchProcessor
from xai_methods.base.base_explainer import BaseExplainer
from xai_methods.explainer_registry import ExplainerRegistry
from xai_methods.implementations.grand_cam_explainer import GradCamExplainer


class XAIFactory:
    """Factory for creating XAI explainers using the registry"""

    def __init__(self):
        self.registry = ExplainerRegistry.get_instance()
        self._register_default_explainers()

    def _register_default_explainers(self):
        """Register default explainer implementations"""
        self.registry.register("gradcam", GradCamExplainer)

    def create_explainer(self, name: str, model: nn.Module,
                         batch_processor: BatchProcessor = None, **kwargs) -> BaseExplainer:
        """
        Create an explainer instance

        Args:
            name: Name of the explainer to create
            model: PyTorch model to explain
            **kwargs: Additional arguments for explainer initialization

        Returns:
            Configured explainer instance
            :param name: todo
            :param model:
            :param batch_processor:
        """
        # Factory ist verantwortlich für vollständige Konfiguration
        if batch_processor is None:
            batch_processor = DirectBatchProcessor()

        explainer_class = self.registry.get(name)
        return explainer_class(model, batch_processor, **kwargs)

    def list_available_explainers(self) -> List[str]:
        """List all available explainer types"""
        return self.registry.list_available()