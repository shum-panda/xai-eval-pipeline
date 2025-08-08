from dataclasses import dataclass
from typing import Any, Dict

from src.pipe.moduls.xai_methods.base.base_xai_config import BaseXAIConfig


@dataclass
class GuidedBackpropConfig(BaseXAIConfig):
    """
    Configuration for Guided Backpropagation (currently no tunable parameters).
    """

    def get_defaults(self) -> Dict[str, Any]:
        return {}

    def validate(self) -> None:
        pass  # Nothing to validate for now
