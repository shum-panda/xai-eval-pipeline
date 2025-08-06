import logging
from typing import List

from src.pipeline.pipeline_moduls.models.base.xai_model import XAIModel
from src.pipeline.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline.pipeline_moduls.xai_methods.explainer_registry import (
    ExplainerRegistry,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.grad_cam.grand_cam_explainer import (
    GradCamExplainer,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.guided_backprop.guided_backprop_expl import (
    GuidedBackpropExplainer,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.integrated_gradients.integrated_gradients_explainer import (
    IntegratedGradientsExplainer,
)
from src.pipeline.pipeline_moduls.xai_methods.impl.score_cam.score_cam_explainer import (
    ScoreCamExplainer,
)


class XAIFactory:
    """
    Factory class for creating XAI explainer instances using a central registry.

    This class encapsulates the logic to register and retrieve explainer classes,
    providing a clean interface for instantiating configured explainers.
    """

    def __init__(self) -> None:
        """Initializes the factory with a shared explainer registry and registers
        default explainers."""
        self._logger = logging.getLogger(__name__)
        self._registry: ExplainerRegistry = ExplainerRegistry.get_instance()
        self._register_default_explainers()

    @property
    def registry(self) -> ExplainerRegistry:
        """
        Returns the internal registry used to manage explainer classes.

        Returns:
            ExplainerRegistry: The registry instance.
        """
        return self._registry

    def _register_default_explainers(self) -> None:
        """
        Registers the default set of explainers supported by the system.
        Extend this method to add more explainers.
        """
        self.registry.register("grad_cam", GradCamExplainer)
        self.registry.register("integrated_gradients", IntegratedGradientsExplainer)
        self.registry.register("score_cam", ScoreCamExplainer)
        self.registry.register("guided_backprop", GuidedBackpropExplainer)

    def create_explainer(
        self,
        name: str,
        model: XAIModel,
        use_defaults: bool,
        **kwargs: object,
    ) -> BaseExplainer:
        """
        Creates and returns a configured explainer instance.

        Args:
            name (str): The name of the explainer to create (must be registered).
            model (XAIModel): XAIModel got expected
            use_defaults (bool): Whether to use the explainer's default configuration.
            **kwargs: Additional arguments passed to the explainer constructor.

        Returns:
            BaseExplainer: A configured instance of the selected explainer.
        """
        explainer_class = self.registry.get(name)
        self._logger.debug(f"Explainer was found: {explainer_class}")
        return explainer_class(model, use_defaults, **kwargs)

    def list_available_explainers(self) -> List[str]:
        """
        Lists all available explainer names currently registered.

        Returns:
            List[str]: A list of registered explainer names.
        """
        return self.registry.list_available()
