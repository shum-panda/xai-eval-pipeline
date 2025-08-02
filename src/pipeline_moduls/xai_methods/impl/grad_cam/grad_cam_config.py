from dataclasses import dataclass
from typing import Any, Dict, List, Union

from sympy import false

from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig


@dataclass
class GradCAMConfig(BaseXAIConfig):
    """
    Configuration class specific to the GradCAM explainability method.

    Attributes:
        target_layer (Union[str, int]):
            Layer to target for GradCAM.
            Default is -1, which triggers automatic detection of the last
            convolutional layer.
        relu_attributions (bool):
            Whether to apply ReLU to the attribution maps.
        interpolate_mode (str):
            Interpolation mode used when resizing the attribution maps.
            Typical values are 'bilinear', 'nearest', etc.
    """

    target_layer: Union[str, int]= None
    relu_attributions: bool= None
    interpolate_mode: str= None

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """
        Returns the default configuration parameters for GradCAM.

        Returns:
            Dict[str, Any]: Dictionary of default parameter values.
        """
        return {
            "target_layer": -1,
            "relu_attributions": True,
            "interpolate_mode": "bilinear",
        }

    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns a list of required parameters which must not be None.

        Returns:
            List[str]: Names of required parameters.
        """
        return ["target_layer", "relu_attributions", "interpolate_mode"]

    @classmethod
    def get_valid_interpolate_modes(cls) -> List[str]:
        """
        Provides the list of valid interpolation modes for resizing attributions.

        Returns:
            List[str]: Valid interpolation mode strings.
        """
        return ["nearest", "linear", "bilinear", "bicubic", "trilinear"]

    def validate(self) -> None:
        """
        Validates the configuration parameters for logical consistency.

        Raises:
            ValueError: If 'interpolate_mode' is not in the list of valid modes.
        """
        for field in self.get_required_params():
            if getattr(self, field, None) is None:
                raise ValueError(f"Required parameter '{field}' is missing.")

        if self.interpolate_mode not in self.get_valid_interpolate_modes():
            raise ValueError(
                f"Invalid interpolate_mode: '{self.interpolate_mode}'. "
                f"Allowed values: {self.get_valid_interpolate_modes()}"
            )
