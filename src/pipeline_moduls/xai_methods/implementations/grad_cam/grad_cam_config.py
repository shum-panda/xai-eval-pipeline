from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig


@dataclass
class GradCAMConfig(BaseXAIConfig):
    """GradCAM-spezifische Konfiguration"""

    target_layer: Union[str, int] = -1  # -1 = auto-detect letzter conv layer
    relu_attributions: bool = True
    interpolate_mode: str = "bilinear"

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Hole Standard-Parameter"""
        return {
            "target_layer": -1,
            "relu_attributions": True,
            "interpolate_mode": "bilinear",
        }

    @classmethod
    def get_required_params(cls) -> List[str]:
        """Hole erforderliche Parameter (die nicht None sein dürfen)"""
        return ["target_layer", "relu_attributions", "interpolate_mode"]

    @classmethod
    def get_valid_interpolate_modes(cls) -> List[str]:
        """Zusätzliche Validation-Helper"""
        return ["nearest", "linear", "bilinear", "bicubic", "trilinear"]

    def validate(self) -> None:
        """Logisch-semantische Validierung der Parameter"""
        if self.interpolate_mode not in self.get_valid_interpolate_modes():
            raise ValueError(
                f"Invalid interpolate_mode: '{self.interpolate_mode}'. "
                f"Allowed values: {self.get_valid_interpolate_modes()}"
            )
