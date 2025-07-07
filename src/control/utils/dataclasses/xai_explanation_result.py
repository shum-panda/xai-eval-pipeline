from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

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

    # Optional fields (with defaults)
    bbox: Optional[torch.Tensor] = None
    bbox_info: Optional[Dict[str, Any]] = None
    dataset_label: Optional[str] = None  # e.g., human-readable class label

    predicted_class_name: Optional[str] = None  # human-readable class name
    true_label: Optional[int] = None
    true_label_name: Optional[str] = None
    prediction_correct: bool = False
    prediction_confidence: Optional[float] = None  # e.g., softmax prob for predicted class
    topk_predictions: Optional[List[int]] = None  # list of top-k predicted class indices
    topk_confidences: Optional[List[float]] = None  # confidences for top-k predictions

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
        data = asdict(self)
        if isinstance(data["image_path"], Path):
            data["image_path"] = str(data["image_path"])
        data["image"] = f"<Tensor shape={tuple(self.image.shape)}>"
        data["attribution"] = f"<Tensor shape={tuple(self.attribution.shape)}>"
        if isinstance(data.get("bbox"), torch.Tensor):
            data["bbox"] = f"<Tensor shape={tuple(self.bbox.shape)}>"
        return data

    @staticmethod
    def from_dict(d: Dict) -> "XAIExplanationResult":
        return XAIExplanationResult(
            image=None,
            attribution=None,
            **{k: v for k, v in d.items() if k not in ["image", "attribution"]}
        )
