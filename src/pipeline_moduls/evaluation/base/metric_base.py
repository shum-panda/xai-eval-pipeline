# metric_base.py
from abc import ABC, abstractmethod
from typing import Any

import torch


class MetricBase(ABC):
    """Abstract base class for all XAI evaluation metrics."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def calculate(self, heatmap: torch.Tensor, ground_truth: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
