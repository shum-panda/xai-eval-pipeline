from dataclasses import dataclass
from typing import Any, Dict, Union

from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig


@dataclass
class ScoreCAMConfig(BaseXAIConfig):
    """
    Configuration class for ScoreCAM.
    """

    target_layer: Union[str, int] = None

    def get_defaults(self) -> Dict[str, Any]:
        """
        Returns default configuration values.

        Returns:
            Dict[str, Any]: Dictionary of default values.
        """
        return {
            "target_layer": -1,
        }

    def validate(self) -> None:
        """
        Validates the current configuration.

        Raises:
            TypeError: If target_layer is neither int nor str.
        """
        if not isinstance(self.target_layer, (int, str)):
            raise TypeError(
                f"'target_layer' must be str or int, but got {type(self.target_layer)}"
            )
