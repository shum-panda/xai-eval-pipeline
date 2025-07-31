from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BaseXAIConfig(ABC):
    use_defaults: bool = True  # <-- zentrales Pflichtfeld mit Defaultwert

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

    @abstractmethod
    def validate(self) -> None:
        """Perform internal validation logic."""
        pass
