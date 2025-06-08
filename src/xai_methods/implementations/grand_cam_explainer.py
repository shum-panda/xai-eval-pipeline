import logging

import torch

from xai_methods.MemoryManagement.batch_processor import BatchProcessor
from xai_methods.MemoryManagement import DirectBatchProcessor


class GradCamExplainer:
    """
    GradCAM Explainer ohne zirkuläre Abhängigkeiten.

    Verwendet Strategy Pattern für Memory Management.
    """

    def __init__(self,
                 model,
                 layer: int = -1,
                 relu_attributions: bool = True,
                 interpolate_mode: str = 'bilinear',
                 batch_processor: BatchProcessor = None):

        self.model = model
        self.layer = layer
        self.relu_attributions = relu_attributions
        self.interpolate_mode = interpolate_mode

        # Strategy Pattern: Batch Processor injizieren
        self.batch_processor = batch_processor or DirectBatchProcessor()

        # GradCAM Setup (vereinfacht)
        self.target_layer = self._select_target_layer(model, layer)

        self.logger = logging.getLogger(__name__)

    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        Hauptmethode - delegiert an Strategy.

        KEINE zirkuläre Abhängigkeit: Processor bekommt nur Callback!
        """

        # Callback-Funktion erstellen (kein self-Referenz)
        def explain_batch(batch_images: torch.Tensor) -> torch.Tensor:
            return self._compute_gradcam_attributions(batch_images)

        # An Strategy delegieren
        return self.batch_processor.process_batch(images, explain_batch)

    def _compute_gradcam_attributions(self, images: torch.Tensor) -> torch.Tensor:
        """Interne GradCAM-Berechnung."""
        # Vereinfachte GradCAM Implementation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Hier würde die echte GradCAM-Berechnung stehen
        # Für Demo: Dummy-Attributions
        attributions = torch.randn_like(images)

        if self.relu_attributions:
            attributions = torch.relu(attributions)

        return attributions

    def _select_target_layer(self, model, layer_idx):
        """Layer-Selektion (vereinfacht)."""
        layers = list(model.children())
        return layers[layer_idx] if layers else model

    def set_batch_processor(self, processor: BatchProcessor) -> None:
        """Ermöglicht Runtime-Wechsel der Strategy."""
        self.batch_processor = processor
        self.logger.info(f"Batch processor changed to: {type(processor).__name__}")
