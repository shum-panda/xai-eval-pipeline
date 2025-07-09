from typing import Protocol

import torch


class ExplainerCallable(Protocol):
    """Protocol für Explainer-Funktionen."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Erklärt einen Batch von Bildern."""
        ...
