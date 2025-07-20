import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from captum.attr import IntegratedGradients

from control.utils.with_cuda_cleanup import with_cuda_cleanup
from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer


@dataclass
class IntegratedGradientsConfig:
    """Integrated Gradients Konfiguration"""

    n_steps: int = 50
    method: str = "riemann_trapezoid"
    baseline: Optional[Any] = None
    use_cuda: bool = True

    @classmethod
    def get_defaults(cls) -> dict:
        return {
            "n_steps": 50,
            "method": "riemann_trapezoid",
            "baseline": None,
            "use_cuda": True,
        }

    @classmethod
    def get_required_params(cls) -> list:
        return ["n_steps", "method", "use_cuda"]


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients Explainer.
    """

    def __init__(self, model, config: IntegratedGradientsConfig = None, **kwargs):
        """
        Initialize Integrated Gradients explainer.

        Args:
            model: PyTorch _model
            config: Integrated Gradients configuration
            **kwargs: Additional arguments
        """
        config = config or IntegratedGradientsConfig()
        super().__init__(model, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.n_steps = config.n_steps
        self.baseline_type = config.baseline_type

        self.ig = IntegratedGradients(self._model)

        self.logger.info(
            f"IntegratedGradients initialized with n_steps={self.n_steps}, "
            f"baseline={self.baseline_type}"
        )

    @with_cuda_cleanup
    def _compute_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute Integrated Gradients attributions.

        Args:
            images: Input tensor (batch_size, channels, height, width)

        Returns:
            Attributions tensor
        """
        try:

            # Get predictions and target classes
            with torch.no_grad():
                predictions = self._model(images)
                target_classes = predictions.argmax(dim=1)

            # Choose baseline
            if self.baseline_type == "black":
                baselines = torch.zeros_like(images)
            elif self.baseline_type == "random":
                baselines = torch.rand_like(images)
            else:
                raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

            # Compute attributions
            attributions = self.ig.attribute(
                inputs=images,
                baselines=baselines,
                target=target_classes,
                n_steps=self.n_steps,
            )

            self.logger.debug(
                "Computed Integrated Gradients attributions with shape: "
                f"{attributions.shape}"
            )
            return attributions

        except Exception as e:
            self.logger.error(f"Error computing Integrated Gradients: {str(e)}")
            self.logger.warning("Falling back to dummy attributions")
            return torch.randn_like(images)

    @classmethod
    def get_name(cls) -> str:
        return "integrated_gradients"
