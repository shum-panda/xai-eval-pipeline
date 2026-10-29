from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from src.pipe.moduls.models.base.xai_model import XAIModel


class TinyCNN(XAIModel):
    """Small CNN with the same interface as the pipeline models, runs on CPU."""

    def __init__(self, model_name: str = "tiny_cnn", **kwargs: Any) -> None:
        super().__init__(model_name)
        torch.manual_seed(0)
        self._device = torch.device("cpu")
        self._net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 5),
        ).eval()

    def get_conv_layers(self) -> List[str]:
        return [n for n, m in self._net.named_modules() if isinstance(m, nn.Conv2d)]

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        return dict(self._net.named_modules())[layer_name]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model_name}

    @property
    def pytorch_model(self) -> nn.Module:
        return self._net

    def get_predictions(self, images: Tensor) -> Tensor:
        return self._net(images)
