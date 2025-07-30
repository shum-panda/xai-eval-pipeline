from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


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
    image: Optional[torch.Tensor]
    image_name: str
    image_path: Union[str, Path]
    has_bbox: bool
    predicted_class: int
    attribution: torch.Tensor
    attribution_path: Optional[str] = None

    # Optional fields (with defaults)
    bbox: Optional[torch.Tensor] = None
    bbox_info: Optional[Dict[str, Any]] = None
    dataset_label: Optional[str] = None  # Human-readable class label from dataset

    predicted_class_name: Optional[str] = None  # Human-readable model prediction
    true_label: Optional[int] = None
    true_label_name: Optional[str] = None
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the explanation result to a serializable dictionary.

        Tensors are represented by a placeholder string with their shape.
        Paths are converted to strings. All other values are passed through.

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

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "XAIExplanationResult":
        """
        Create an XAIExplanationResult instance from a dictionary.

        The fields `image` and `attribution` are intentionally set to None,
        since tensor deserialization must be handled externally.

        Args:
            d (dict): A dictionary, typically from a serialized JSON or logging source.

        Returns:
            XAIExplanationResult: The reconstructed explanation result object.
        """
        return XAIExplanationResult(
            image=None,
            attribution=torch.zeros(),
            **{k: v for k, v in d.items() if k not in ["image", "attribution"]},
        )
