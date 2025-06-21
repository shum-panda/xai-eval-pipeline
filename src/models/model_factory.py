import logging
import threading
from typing import Optional

import torch

from models.model_interface import ModelInterface


class ModelFactory:
    """Singleton factory for loading models - one model at a time"""

    _instance: Optional['ModelFactory'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._current_model: Optional[ModelInterface] = None
            self.logger = logging.getLogger(__name__)
            self._initialized = True

    @classmethod
    def get_instance(cls) -> 'ModelFactory':
        """Get the singleton factory instance"""
        return cls()

    def load_model(self, model_name: str, architecture: str = None) -> ModelInterface:
        """
        Load a model from PyTorch Hub

        Args:
            model_name: Name to identify the model
            architecture: PyTorch Hub architecture name (defaults to model_name)

        Returns:
            ModelInterface for the loaded model

        Raises:
            RuntimeError: If model loading fails
            ValueError: If architecture not supported
        """
        if architecture is None:
            architecture = model_name

        # Unload previous model if exists
        if self._current_model is not None:
            self._unload_current_model()

        try:
            self.logger.info(f"Loading model '{model_name}' (architecture: {architecture})")

            # Load from PyTorch Hub with fail-fast
            model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                architecture,
                pretrained=True,
                verbose=False
            )

            # Create interface
            self._current_model = ModelInterface(model, model_name)

            # Log success
            info = self._current_model.get_model_info()
            self.logger.info(f"Successfully loaded {info['name']} with {info['total_parameters']:,} parameters")

            return self._current_model

        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def _unload_current_model(self):
        """Unload current model to free memory"""
        if self._current_model is not None:
            model_name = self._current_model.model_name

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._current_model = None
            self.logger.info(f"Unloaded model '{model_name}'")

    @property
    def current_model(self) -> Optional[ModelInterface]:
        """Get the currently loaded model"""
        return self._current_model

    def has_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self._current_model is not None

    def get_current_model_info(self) -> dict:
        """Get info about current model"""
        if self._current_model is None:
            return {'status': 'no_model_loaded'}

        return self._current_model.get_model_info()
