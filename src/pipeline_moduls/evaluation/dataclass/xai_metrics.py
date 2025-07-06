from dataclasses import dataclass


@dataclass
class XAIMetrics:
    """
    Strukturierte XAI Evaluation Metriken
    """
    # Pointing Game
    pointing_game_hit: bool
    pointing_game_threshold: float

    # IoU Scores
    iou_score: float
    iou_threshold: float

    # Coverage Metrics
    coverage_score: float
    coverage_percentile: float

    # Intersection Metrics
    intersection_area: float
    bbox_area: float
    attribution_area: float

    # Additional Metrics
    precision: float
    recall: float
