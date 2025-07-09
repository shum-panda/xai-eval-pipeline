from pathlib import Path
from unittest.mock import Mock

import torch

from control.xai_orchestrator import XAIOrchestrator


def test_explain_batch_with_mock():
    evaluator = XAIOrchestrator()

    # Set up input batch (korrekt)
    B, C, H, W = 2, 3, 224, 224
    images_tensor = torch.rand(B, C, H, W)
    labels_tensor = torch.tensor([1, 2])
    boxes_list = [torch.rand(1, 4) for _ in range(B)]
    image_paths = [Path(f"/tmp/img_{i}.jpg") for i in range(B)]
    image_names = [f"img_{i}.jpg" for i in range(B)]
    bbox_paths = [Path(f"/tmp/bbox_{i}.json") for i in range(B)]
    labels_int = [1, 2]

    batch = (
        images_tensor,
        labels_tensor,
        boxes_list,
        image_paths,
        image_names,
        bbox_paths,
        labels_int,
    )

    # 🔸 Mock für explainer + Rückgabeobjekt
    mock_explainer = Mock()
    mock_result = Mock()
    mock_result.attributions = torch.rand(B, C, H, W)
    mock_result.predictions = torch.tensor([1, 2])
    mock_result.target_labels = labels_tensor
    mock_explainer.explain.return_value = mock_result

    results = evaluator.explain_batch(batch, mock_explainer)

    assert len(results) == B
    for r in results:
        assert hasattr(r, "predicted_class")
        assert hasattr(r, "attribution")
