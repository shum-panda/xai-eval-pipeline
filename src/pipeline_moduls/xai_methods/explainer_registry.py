import threading
from typing import Dict, List, Optional, Type

from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer


class ExplainerRegistry:
    """
    Singleton registry for managing XAI explainer classes

    Attributes:
        _instance: Singleton instance
        _registry: Dictionary mapping explainer names to classes
    """

    _instance: Optional["ExplainerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ExplainerRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized") or not self._initialized:
            self._registry: Dict[str, Type[BaseExplainer]] = {}
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "ExplainerRegistry":
        """Get the singleton registry instance"""
        return cls()

    def register(
        self, name: Optional[str], explainer_class: Type[BaseExplainer]
    ) -> None:
        """
        Register an explainer class in the registry

        Args:
            name: String identifier for the explainer
            explainer_class: Class that inherits from BaseExplainer
        """
        if not issubclass(explainer_class, BaseExplainer):
            raise TypeError("Explainer class must inherit from BaseExplainer")

        if not name:
            name = explainer_class.get_name()
        self._registry[name] = explainer_class

    def get(self, name: str) -> Type[BaseExplainer]:
        """
        Get an explainer class by name

        Args:
            name: String identifier of the explainer

        Returns:
            The explainer class

        Raises:
            KeyError: If explainer name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Explainer '{name}' not found in registry")
        return self._registry[name]

    def list_available(self) -> List[str]:
        """Return list of all registered explainer names"""
        return list(self._registry.keys())
