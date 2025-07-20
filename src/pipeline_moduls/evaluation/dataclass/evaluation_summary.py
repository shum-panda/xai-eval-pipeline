from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EvaluationSummary:
    """
    Summary einer kompletten Evaluation
    """
    explainer_name: str
    model_name: str
    total_samples: int
    samples_with_bbox: int
    prediction_accuracy: float
    correct_predictions: int
    average_processing_time: float
    total_processing_time: float
    evaluation_timestamp: str

    # Dynamische Metriken
    metric_averages: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = self.__dict__.copy()
        base.update(self.metric_averages)
        return base
