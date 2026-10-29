import unittest
from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from src.pipe.moduls.models.base.xai_model import XAIModel
from src.pipe.moduls.xai_methods.impl.ig.integrated_gradients_explainer import (
    IntegratedGradientsExplainer,
)


class LinearModel(XAIModel):
    """Tiny linear model: for f(x) = w . x with a zero baseline, IG equals x * w."""

    def __init__(self) -> None:
        super().__init__("linear")
        torch.manual_seed(0)
        self._net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2, bias=False))
        self._net.eval()
        self._device = torch.device("cpu")

    def get_conv_layers(self) -> List[str]:
        return []

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        raise ValueError(layer_name)

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "linear"}

    @property
    def pytorch_model(self) -> nn.Module:
        return self._net

    def get_predictions(self, images: Tensor) -> Tensor:
        return self._net(images)


class TestIntegratedGradients(unittest.TestCase):

    def test_attributions_are_input_times_weight(self) -> None:
        model = LinearModel()
        explainer = IntegratedGradientsExplainer(model, use_defaults=True)
        images = torch.rand(1, 3, 4, 4) + 0.5
        target = torch.tensor([1])

        attributions = explainer._compute_attributions(images, target)

        weights = model.pytorch_model[1].weight[1].detach().reshape(1, 3, 4, 4)
        expected = (images * weights).abs().max(dim=1)[0]
        torch.testing.assert_close(attributions, expected, rtol=1e-4, atol=1e-5)

    def test_strong_negative_attribution_is_kept(self) -> None:
        model = LinearModel()
        weights = torch.full((3, 4, 4), 0.1)
        weights[0] = -3.0  # red channel: strongly negative, the others weakly positive
        with torch.no_grad():
            model.pytorch_model[1].weight[1].copy_(weights.flatten())
        explainer = IntegratedGradientsExplainer(model, use_defaults=True)
        images = torch.ones(1, 3, 4, 4)

        attributions = explainer._compute_attributions(images, torch.tensor([1]))

        torch.testing.assert_close(attributions, torch.full((1, 4, 4), 3.0))


if __name__ == "__main__":
    unittest.main()
