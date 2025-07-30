from typing import Any, Dict

from captum.attr import IntegratedGradients  # type: ignore
from torch import Tensor

from src.control.utils.with_cuda_cleanup import with_cuda_cleanup
from src.pipeline_moduls.models.base.interface.xai_model import XAIModel
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from src.pipeline_moduls.xai_methods.impl.integrated_gradients.integrated_gradients_config import (
    IntegratedGradientsConfig,
)


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Explainer class implementing Integrated Gradients using Captum.
    """

    @property
    def parameters(self) -> Dict[str, str]:
        return {
            "n_steps": str(self.n_steps),
            "internal_batch_size": str(self.internal_batch_size),
            "multiply_by_inputs": str(self.multiply_by_inputs),
        }

    def __init__(
        self, model: XAIModel, use_defaults: bool = True, **kwargs: object
    ) -> None:
        self.integrated_gradients = None
        self.n_steps = 50
        self.internal_batch_size = None
        self.multiply_by_inputs = True
        super().__init__(model, use_defaults, **kwargs)

    @with_cuda_cleanup
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        if self.integrated_gradients is None:
            raise RuntimeError("IntegratedGradients has not been initialized.")

        try:
            attributions = self.integrated_gradients.attribute(
                inputs=images,
                target=target_classes,
                n_steps=self.n_steps,
                internal_batch_size=self.internal_batch_size,
                return_convergence_delta=False,
            )

            if self.multiply_by_inputs:
                attributions = attributions * images

            self._logger.debug(
                f"Computed Integrated Gradients with shape: {attributions.shape}"
            )
            return attributions.detach().cpu()
        except Exception as e:
            self._logger.error(f"Error computing Integrated Gradients: {str(e)}")
            raise

    def check_input(self, **kwargs: Any) -> BaseXAIConfig:
        try:
            config = IntegratedGradientsConfig(**kwargs)
            config.validate()
        except (TypeError, ValueError) as e:
            self._logger.error(f"Invalid config: {e}")
            raise
        return config

    def _setup_with_validated_params(self, config: BaseXAIConfig) -> None:
        if not isinstance(config, IntegratedGradientsConfig):
            raise TypeError(
                f"Expected IntegratedGradientsConfig, got {type(config).__name__}"
            )

        self.n_steps = config.n_steps
        self.internal_batch_size = config.internal_batch_size
        self.multiply_by_inputs = config.multiply_by_inputs

        self.integrated_gradients = IntegratedGradients(self._model.pytorch_model)

        self._logger.info(
            f"IntegratedGradients setup complete with n_steps: {self.n_steps}, "
            f"internal_batch_size: {self.internal_batch_size}, "
            f"multiply_by_inputs: {self.multiply_by_inputs}"
        )

    @classmethod
    def get_name(cls) -> str:
        return "integrated_gradients"
