from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    def __init__(self, model, **kwargs):
        self.model = model
        # Subklassen verarbeiten ihre kwargs hier

    @abstractmethod
    def explain(self, images):  # ← Batch von Bildern
        """
        Args:
            images: Batch von Bildern (z.B. torch.Tensor [N, C, H, W])

        Returns:
            heatmaps: Batch von Heatmaps [N, H, W]
            predictions: Batch von Predictions (List oder Array)
        """
        pass