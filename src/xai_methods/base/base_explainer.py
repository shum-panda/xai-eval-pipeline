import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch

from xai_methods.MemoryManagement.base.batch_processor import BatchProcessor

class BaseExplainer(ABC):

    def __init__(self, model, batch_processor: BatchProcessor=None, **kwargs):
        self.logger = None
        if batch_processor is None:
            raise ValueError("BatchProcessor is required. Use DirectBatchProcessor() for no batching.")

        self.model = model

        # Strategy Pattern: Batch Processor injizieren
        self.batch_processor = batch_processor
        self.logger= logging.getLogger(__name__) #fallbacklooger

    def set_batch_processor(self, processor: BatchProcessor) -> None:
        """Ermöglicht Runtime-Wechsel der Strategy."""
        self.batch_processor = processor
        self.logger.info(f"Batch processor changed to: {type(processor).__name__}")

    def explain(self, images: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            images: Batch von Bildern (z.B. torch.Tensor [N, C, H, W])

        Returns:
            heatmaps: Batch von Heatmaps [N, H, W]
            predictions: Batch von Predictions (List oder Array)

        Hauptmethode - delegiert an Strategy.

        KEINE zirkuläre Abhängigkeit: Processor bekommt nur Callback!
        """

        # Callback-Funktion erstellen (kein self-Referenz)
        def explain_batch(batch_images: torch.Tensor) -> torch.Tensor:
            return self._compute_attributions(batch_images)

        # An Strategy delegieren
        return self.batch_processor.process_batch(images, explain_batch)

    @abstractmethod
    def _compute_attributions(self, batch_images: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Compute attributions for a batch of images.

        This is the method that subclasses must implement.

        Args:
            batch_images: Batch of input images

        Returns:
            Attribution maps for the batch
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name identifier of this explainer"""
        pass

