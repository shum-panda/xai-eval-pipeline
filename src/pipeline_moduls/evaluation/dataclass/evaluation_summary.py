from dataclasses import dataclass


@dataclass
class EvaluationSummary:
    """
    Summary einer kompletten Evaluation
    """

    # Basic Info
    explainer_name: str
    model_name: str
    total_samples: int
    samples_with_bbox: int

    # Prediction Metrics
    prediction_accuracy: float
    correct_predictions: int

    # XAI Metrics (Averages)
    pointing_game_score: float
    average_iou: float
    average_coverage: float
    average_precision: float
    average_recall: float

    # Performance
    average_processing_time: float
    total_processing_time: float

    # Timestamps
    evaluation_timestamp: str
