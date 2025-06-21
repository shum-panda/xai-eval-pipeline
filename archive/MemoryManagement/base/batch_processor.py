from abc import ABC, abstractmethod

import torch

from archive.MemoryManagement.base.explainer_callable import ExplainerCallable


class BatchProcessor(ABC):
    """Abstrakte Basis für Batch Processing Strategien."""

    @abstractmethod
    def process_batch(self,
                      images: torch.Tensor,
                      explain_fn: ExplainerCallable) -> torch.Tensor:
        """
        Verarbeitet Bilder mit der gegebenen Explain-Funktion.

        Args:
            images: Input-Bilder
            explain_fn: Funktion die Bilder erklärt

        Returns:
            Erklärte Attributions
        """
        pass