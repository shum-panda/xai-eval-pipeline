from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

import torch


@dataclass
class XAIExplanationResult:
    """
    Strukturiertes Ergebnis einer XAI Explanation
    """
    image:torch.Tensor
    image_name: str
    image_path: Path

    # Model Prediction
    predicted_class: int
    true_label: Optional[int]

    # XAI Explanation
    attribution: torch.Tensor
    explainer_result: Any
    explainer_name: str
    prediction_correct:bool

    # Dataset Info
    has_bbox: bool
    bbox: torch.Tensor
    bbox_info: Optional[Dict]

    # Metadata
    model_name: str
    processing_time: float

