from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BaseXAIConfig(ABC):
    use_defaults: bool

    def __post_init__(self):
        if self.use_defaults:
            defaults = self.get_defaults()
            for key, value in defaults.items():
                if getattr(self, key, None) is None:
                    setattr(self, key, value)
        self.validate()

    @classmethod
    @abstractmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Return a dictionary of default values."""
        pass

    @classmethod
    def get_options(cls) -> Dict[str, List[Any]]:
        """
        Allowed values for parameters with a fixed set of choices.

        Used by tools such as the pipeline UI to offer a selection instead of
        free text. Parameters not listed accept any value of their type.
        """
        return {}

    @abstractmethod
    def validate(self) -> None:
        """Perform internal validation logic."""
        pass
