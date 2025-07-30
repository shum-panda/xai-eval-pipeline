from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.ressource_management.attribution_reference import AttributionReference


@dataclass
class XAIExplanationResult:
    """Structured container for the result of a single XAI explanation applied to one
    image.

    This data structure holds all relevant inputs, model outputs, explanation artifacts,
    and optional evaluation metrics or metadata. It is designed to support standardized
    logging, analysis, and visualization of XAI results in image classification
    pipelines.
    """

    # Required fields (no defaults)
    image_name: str
    image_path: Union[str, Path]
    has_bbox: bool
    attribution: AttributionReference
    predicted_class: int
    predicted_class_name: Optional[str] = None  # Human-readable _model prediction
    true_label: Optional[int] = None
    true_label_name: Optional[str] = None
    attribution_path: Optional[str] = None

    # Optional fields (with defaults)
    bbox: Optional[torch.Tensor] = None

    prediction_correct: bool = False
    prediction_confidence: Optional[float] = None
    predicted_class_before_transform: Optional[int] = (
        None  # Original class index before mapping
    )
    topk_predictions: Optional[List[int]] = None  # Top-k predicted class indices
    topk_confidences: Optional[List[float]] = None  # Confidence scores for top-k

    explainer_result: Optional[Any] = None  # Raw output returned by explainer
    explainer_name: str = ""

    model_name: str = ""
    model_version: Optional[str] = None
    explainer_params: Optional[Dict[str, Any]] = (
        None  # Parameters used by the explainer
    )
    processing_time: float = 0.0  # Runtime in seconds
    timestamp: Optional[str] = None  # ISO timestamp for logging or provenance

    @property
    def attribution_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the actual attribution tensor, handling both direct tensors and references.
        """
        return self.attribution.attribution

    def clear_attribution_cache(self):
        """Clear attribution cache if using reference"""
        if isinstance(self.attribution, AttributionReference):
            self.attribution.clear_cache()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the explanation result to a serializable dictionary.
        Tensors are represented by a placeholder string with their shape.
        Paths are converted to strings. Fields with `None` are excluded.

        Returns:
            dict: A dictionary representation of the explanation result.
        """
        data = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                continue  # <-- SKIP None entries
            if isinstance(value, torch.Tensor):
                data[field] = f"<Tensor shape={tuple(value.shape)}>"
            elif isinstance(value, Path):
                data[field] = str(value)
            else:
                data[field] = value
        return data
