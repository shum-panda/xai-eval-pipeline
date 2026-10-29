import unittest

import torch
from tiny_models import TinyCNN

from src.pipe.moduls.models.xai_model_factory import XAIModelFactory


class TestModelFactory(unittest.TestCase):

    def setUp(self) -> None:
        self.factory = XAIModelFactory()
        self.factory.register_model("tiny_cnn_test", TinyCNN)

    def tearDown(self) -> None:
        self.factory.unload_current_model()

    def test_builtin_models_are_registered(self) -> None:
        available = self.factory.list_available()
        for name in ("resnet18", "resnet34", "resnet50", "vgg16"):
            self.assertIn(name, available)

    def test_custom_model_is_created_with_orchestrator_kwargs(self) -> None:
        # Same keyword arguments the orchestrator passes from the config
        model = self.factory.create(
            "tiny_cnn_test", seed=1, pretrained=True, device=torch.device("cpu")
        )

        self.assertIsInstance(model, TinyCNN)
        self.assertEqual(model.model_name, "tiny_cnn_test")
        self.assertIs(self.factory.get_current_model(), model)

    def test_creating_a_new_model_replaces_the_current_one(self) -> None:
        first = self.factory.create("tiny_cnn_test")
        second = self.factory.create("tiny_cnn_test")

        self.assertIsNot(first, second)
        self.assertIs(self.factory.get_current_model(), second)

    def test_unload_frees_the_current_model(self) -> None:
        self.factory.create("tiny_cnn_test")
        self.factory.unload_current_model()
        self.assertFalse(self.factory.has_model_loaded())

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.factory.create("no_such_model")

    def test_non_model_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.factory.register_model("bad", dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
