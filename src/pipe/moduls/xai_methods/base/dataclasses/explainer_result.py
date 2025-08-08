from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor


@dataclass
class ExplainerResult:
    """Clean result object containing everything needed for evaluation"""

    attributions: Tensor
    probabilities: Tensor
    predictions: Tensor
    confidence: Tensor
    target_labels: Tensor
    topk_predictions: Optional[Tensor] = None  # shape: (B, k)
    topk_confidences: Optional[Tensor] = None  # shape: (B, k)

    @property
    def correct_predictions(self) -> Tensor:
        """Boolean tensor indicating correct predictions"""
        return self.predictions == self.target_labels

    @property
    def accuracy(self) -> float:
        """Overall accuracy as float"""
        return self.correct_predictions.float().mean().item()

    @property
    def num_correct(self) -> int:
        """Number of correct predictions"""
        return int(self.correct_predictions.sum().item())

    @property
    def num_total(self) -> int:
        """Total number of predictions"""
        return len(self.target_labels)

    def to_dict(self) -> Dict[str, Any]:
        """
        Create a dict summary for logging or export,
        avoiding large tensor data but including shapes and metrics.
        """

        def safe_shape(tensor):
            return tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None

        return {
            "accuracy": self.accuracy,
            "correct": self.num_correct,
            "total": self.num_total,
            "attribution_shape": safe_shape(self.attributions),
            "probabilities_shape": safe_shape(self.probabilities),
            "predictions_shape": safe_shape(self.predictions),
            "confidence_shape": safe_shape(self.confidence),
            "target_labels_shape": safe_shape(self.target_labels),
            "topk_predictions_shape": safe_shape(self.topk_predictions),
            "topk_confidences_shape": safe_shape(self.topk_confidences),
        }
