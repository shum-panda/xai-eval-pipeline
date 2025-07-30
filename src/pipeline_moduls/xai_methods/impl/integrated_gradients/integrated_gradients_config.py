from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig


@dataclass
class IntegratedGradientsConfig(BaseXAIConfig):
    """
    Configuration class for Integrated Gradients.

    Attributes:
        n_steps (int): Number of steps in the Riemann approximation.
        internal_batch_size (Optional[int]): Batch size for internal steps.
        multiply_by_inputs (bool): Whether to multiply attributions by the input.
    """

    n_steps: int = 50
    internal_batch_size: Optional[int] = None
    multiply_by_inputs: bool = True

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        return {
            "n_steps": 50,
            "internal_batch_size": None,
            "multiply_by_inputs": True,
        }

    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["n_steps", "multiply_by_inputs"]

    def validate(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.internal_batch_size is not None and self.internal_batch_size <= 0:
            raise ValueError("internal_batch_size must be None or positive.")
