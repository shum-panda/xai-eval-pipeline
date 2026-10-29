import unittest
from typing import Any, Dict

import torch
from tiny_models import TinyCNN
from torch import Tensor

from src.pipe.moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipe.moduls.xai_methods.xai_factory import XAIFactory

BUILTIN_EXPLAINERS = [
    "grad_cam",
    "score_cam",
    "guided_backprop",
    "integrated_gradients",
]


class TestBuiltinExplainers(unittest.TestCase):
    """Every built-in method works end to end through the factory."""

    def setUp(self) -> None:
        torch.manual_seed(1)
        self.model = TinyCNN()
        self.images = torch.rand(2, 3, 32, 32)
        self.labels = torch.tensor([1, 3])

    def test_all_builtin_explainers_are_registered(self) -> None:
        available = XAIFactory().list_available_explainers()
        for name in BUILTIN_EXPLAINERS:
            self.assertIn(name, available)

    def test_each_explainer_returns_one_2d_heatmap_per_image(self) -> None:
        for name in BUILTIN_EXPLAINERS:
            with self.subTest(explainer=name):
                explainer = XAIFactory().create_explainer(
                    name, self.model, use_defaults=True
                )
                result = explainer.explain(self.images, self.labels, top_k=3)

                self.assertEqual(result.attributions.shape, (2, 32, 32))
                self.assertTrue(torch.isfinite(result.attributions).all())
                self.assertGreaterEqual(float(result.attributions.min()), 0.0)
                self.assertEqual(result.topk_predictions.shape, (2, 3))
                torch.testing.assert_close(
                    result.probabilities.sum(dim=1), torch.ones(2)
                )

    def test_explanations_are_repeatable(self) -> None:
        for name in ("grad_cam", "guided_backprop", "integrated_gradients"):
            with self.subTest(explainer=name):
                explainer = XAIFactory().create_explainer(
                    name, self.model, use_defaults=True
                )
                first = explainer.explain(self.images, self.labels, top_k=1)
                second = explainer.explain(self.images, self.labels, top_k=1)
                torch.testing.assert_close(first.attributions, second.attributions)

    def test_unknown_explainer_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            XAIFactory().create_explainer("no_such_method", self.model, True)


class ConstantExplainer(BaseExplainer):
    """Minimal custom method: shows what it takes to add a new XAI method."""

    @property
    def parameters(self) -> Dict[str, str]:
        return {}

    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        return torch.ones(images.shape[0], *images.shape[2:])

    def _check_input(self, **kwargs: Any) -> Any:
        return None

    def _setup_with_validated_params(self, config: Any) -> None:
        pass

    @classmethod
    def get_name(cls) -> str:
        return "constant_test_explainer"


class TestCustomExplainerRegistration(unittest.TestCase):

    def test_registered_custom_explainer_can_be_created_by_name(self) -> None:
        factory = XAIFactory()
        factory.registry.register(None, ConstantExplainer)

        explainer = factory.create_explainer(
            "constant_test_explainer", TinyCNN(), use_defaults=True
        )
        result = explainer.explain(torch.rand(1, 3, 8, 8), torch.tensor([0]), top_k=1)

        torch.testing.assert_close(result.attributions, torch.ones(1, 8, 8))

    def test_non_explainer_class_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            XAIFactory().registry.register("bad", dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
