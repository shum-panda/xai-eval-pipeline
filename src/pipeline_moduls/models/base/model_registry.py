import logging
import threading
from typing import Dict, List, Optional, Type

from pipeline_moduls.models.base.interface.xai_model import XAIModel


class ModelRegistry:
    """Singleton registry for XAI model classes (consistent with ExplainerRegistry)"""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        if ModelRegistry._instance is not None:
            raise RuntimeError("ModelRegistry is a singleton. Use get_instance()")

        self._registry: Dict[str, Type[XAIModel]] = {}
        self.logger = logging.getLogger(__name__)

        # Auto-register built-in model types
        self._register_builtin_models()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Get the singleton registry instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, model_class: Type[XAIModel]) -> None:
        """Register a model class with a name

        Args:
            name: Model identifier (e.g., 'resnet50', 'my_custom_net')
            model_class: XAIModel subclass to register
        """
        if not issubclass(model_class, XAIModel):
            raise ValueError(
                f"Model class must inherit from XAIModel, got {model_class}"
            )

        if name in self._registry:
            self.logger.warning(f"Overwriting existing model registration for '{name}'")

        self._registry[name] = model_class
        self.logger.info(f"Registered model '{name}' -> {model_class.__name__}")

    def get(self, name: str) -> Type[XAIModel]:
        """Get a registered model class

        Args:
            name: Model name

        Returns:
            The registered XAIModel class

        Raises:
            ValueError: If model name not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Model '{name}' not registered. Available models: {available}"
            )

        return self._registry[name]

    def list_available(self) -> List[str]:
        """Get list of all registered model names"""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a model name is registered"""
        return name in self._registry

    def unregister(self, name: str) -> None:
        """Unregister a model

        Args:
            name: Model name to unregister

        Raises:
            ValueError: If model not registered
        """
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not registered")

        del self._registry[name]
        self.logger.info(f"Unregistered model '{name}'")

    def get_registry_info(self) -> Dict[str, str]:
        """Get information about registered models

        Returns:
            Dict mapping model names to their class names
        """
        return {name: cls.__name__ for name, cls in self._registry.items()}

    def _register_builtin_models(self) -> None:
        """Register built-in model types"""
        # Note: These will be registered when the actual model classes are imported
        # For now, we'll register them when the classes are defined
        pass
