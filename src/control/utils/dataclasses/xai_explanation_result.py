from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class XAIExplanationResult:
    """
    Structured result of a single XAI explanation for one image.
    Includes metadata, model predictions, XAI outputs, and optional evaluation metrics.
    """

    # Required fields (no defaults)
    image: torch.Tensor
    image_name: str
    image_path: Union[str, Path]
    has_bbox: bool
    predicted_class: int
    attribution: torch.Tensor
    attribution_path: Optional[str] = None

    # Optional fields (with defaults)
    bbox: Optional[torch.Tensor] = None
    bbox_info: Optional[Dict[str, Any]] = None
    dataset_label: Optional[str] = None  # e.g., human-readable class label

    predicted_class_name: Optional[str] = None  # human-readable class name
    true_label: Optional[int] = None
    true_label_name: Optional[str] = None
    prediction_correct: bool = False
    prediction_confidence: Optional[float] = None
    topk_predictions: Optional[List[int]] = (
        None  # list of top-k predicted class indices
    )
    topk_confidences: Optional[List[float]] = None  # confidences for top-k predictions
    topk_probabilities: Optional[List[float]] = None

    explainer_result: Optional[Any] = None  # explainer-specific return
    explainer_name: str = ""
    attribution_summary: Optional[float] = None  # e.g., mean absolute attribution value

    explanation_faithfulness: Optional[float] = None
    explanation_sparsity: Optional[float] = None
    explanation_complexity: Optional[float] = None

    model_name: str = ""
    model_version: Optional[str] = None
    explainer_params: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    timestamp: Optional[str] = None  # e.g., ISO time when result was created

    def to_dict(self) -> Dict:
        data = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                data[field] = f"<Tensor shape={tuple(value.shape)}>"
            elif isinstance(value, Path):
                data[field] = str(value)
            else:
                data[field] = value
        return data

    @staticmethod
    def from_dict(d: Dict) -> "XAIExplanationResult":
        return XAIExplanationResult(
            image=None,
            attribution=None,
            **{k: v for k, v in d.items() if k not in ["image", "attribution"]},
        )
