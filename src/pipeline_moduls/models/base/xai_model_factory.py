import logging
import threading
from typing import Dict, List, Optional, Type

from pipeline_moduls.models.base.interface.xai_model import XAIModel
from pipeline_moduls.models.base.model_registry import ModelRegistry
from pipeline_moduls.models.implementation.pytorch_hub_model import PytorchHubModel


class XAIModelFactory:
    """Instance-based factory for creating XAI models with shared registry by default"""

    def __init__(self, registry: ModelRegistry = None):
        """Initialize factory with optional custom registry

        Args:
            registry: Custom ModelRegistry instance. If None, uses singleton registry.
        """
        self.registry = registry or ModelRegistry.get_instance()
        self._current_model: Optional[XAIModel] = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # Ensure builtin models are registered when factory is created
        self._ensure_builtin_models_registered()

    def create(self, name: str, **kwargs) -> XAIModel:
        """Create an XAI _model instance

        Args:
            name: Model name (must be registered in registry)
            **kwargs: Additional arguments passed to the _model constructor

        Returns:
            XAIModel instance

        Raises:
            ValueError: If _model name not registered
            RuntimeError: If _model creation fails
        """
        with self._lock:
            # Memory management - unload previous _model
            if self._current_model is not None:
                self._cleanup_current_model()

            try:
                model_class = self.registry.get(name)

                # Create the _model instance
                self._current_model = model_class(model_name=name, **kwargs)

                self.logger.info(
                    f"Successfully created _model '{name}' of type "
                    f"{model_class.__name__}"
                )
                return self._current_model

            except Exception as e:
                self.logger.error(f"Failed to create _model '{name}': {str(e)}")
                raise RuntimeError(f"Model creation failed: {str(e)}") from e

    def get_current_model(self) -> Optional[XAIModel]:
        """Get the currently loaded _model"""
        return self._current_model

    def unload_current_model(self) -> None:
        """Unload the current _model and free memory"""
        with self._lock:
            self._cleanup_current_model()

    def has_model_loaded(self) -> bool:
        """Check if a _model is currently loaded"""
        return self._current_model is not None

    def list_available(self) -> List[str]:
        """Get list of all available _model names from the registry"""
        return self.registry.list_available()

    def get_registry_info(self) -> Dict[str, str]:
        """Get information about all registered models"""
        return self.registry.get_registry_info()

    def register_model(self, name: str, model_class: Type[XAIModel]) -> None:
        """Convenience method to register a _model in this factory's registry

        Args:
            name: Model name
            model_class: XAIModel subclass to register
        """
        self.registry.register(name, model_class)

    def is_registered(self, name: str) -> bool:
        """Check if a _model name is registered in this factory's registry"""
        return self.registry.is_registered(name)

    def _cleanup_current_model(self) -> None:
        """Internal method to cleanup current _model and free memory"""
        if self._current_model is not None:
            # Get _model info before cleanup
            model_info = self._current_model.get_model_info()
            model_name = model_info.get("name", "unknown")

            # Cleanup memory
            self._cleanup_memory()
            self._current_model = None

            self.logger.info(f"Unloaded _model '{model_name}'")

    def _cleanup_memory(self) -> None:
        """Cleanup memory (can be extended for specific cleanup logic)"""
        import gc

        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _ensure_builtin_models_registered(self) -> None:
        """Ensure builtin models are registered in this factory's registry
        todo should extracted"""
        # Common PyTorch Hub models
        hub_models = [
            "resnet50",
            "vgg16",
        ]

        for model_name in hub_models:
            if not self.registry.is_registered(model_name):
                self.registry.register(model_name, PytorchHubModel)

        self.logger.debug(
            f"Ensured {len(hub_models)} builtin PyTorch Hub models are registered"
        )
