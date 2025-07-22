from dataclasses import dataclass
from typing import Optional

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
        return self.predictions.argmax(dim=1) == self.target_labels

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

    def get_summary(self) -> dict:
        """Get evaluation summary"""
        return {
            "accuracy": self.accuracy,
            "correct": self.num_correct,
            "total": self.num_total,
            "attribution_shape": tuple(self.attributions.shape),
        }
