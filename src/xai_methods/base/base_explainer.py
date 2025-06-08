from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch


class BaseExplainer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, images: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            images: Batch von Bildern (z.B. torch.Tensor [N, C, H, W])

        Returns:
            heatmaps: Batch von Heatmaps [N, H, W]
            predictions: Batch von Predictions (List oder Array)
        """
        pass