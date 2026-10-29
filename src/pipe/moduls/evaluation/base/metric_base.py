from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


class MetricBase(ABC):
    """
    Abstract base class for all XAI evaluation metrics.

    This class defines a common interface for metric implementations
    that evaluate explanation heatmaps against ground truth data.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the metric. Accepts optional keyword arguments for configuration.

        Args:
            **kwargs (Any): Optional metric-specific configuration parameters.
        """
        pass

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """
        Default values of the metric's keyword arguments.

        Used by tools such as the pipeline UI to list a metric's parameters.
        """
        return {}

    @classmethod
    def get_options(cls) -> Dict[str, List[Any]]:
        """Allowed values for parameters with a fixed set of choices."""
        return {}

    @abstractmethod
    def calculate(
        self,
        heatmap: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> float:
        """
        Compute the metric based on the provided heatmap and ground truth.

        Args:
            heatmap (torch.Tensor): The explanation heatmap to be evaluated.
            ground_truth (torch.Tensor): The corresponding ground truth data.

        Returns:
             float: the result value of the metric
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name of the metric.

        Returns:
            str: The name of the metric.
        """
        pass
