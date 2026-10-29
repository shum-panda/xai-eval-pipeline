import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import torch  # noqa: E402

from src.pipe.control.utils.dataclasses.xai_explanation_result import (  # noqa: E402
    XAIExplanationResult,
)
from src.pipe.moduls.visualization.visualisation import Visualiser  # noqa: E402


class TestVisualiserLayout(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def make_result(self, with_bbox: bool) -> XAIExplanationResult:
        torch.manual_seed(0)
        return XAIExplanationResult(
            image=torch.randn(3, 224, 224),
            image_name="ILSVRC2012_val_00000001.JPEG",
            image_path=self.temp_dir / "unused.JPEG",
            has_bbox=with_bbox,
            predicted_class=65,
            attribution=torch.rand(224, 224),
            bbox=torch.tensor([[60.0, 100.0, 160.0, 180.0]]) if with_bbox else None,
            predicted_class_name="sea snake",
            true_label=65,
            true_label_name="sea snake",
            prediction_correct=True,
            explainer_name="grad_cam",
            model_name="resnet50",
        )

    def test_visualization_is_saved_with_bbox(self) -> None:
        visualiser = Visualiser(show=False, save_path=self.temp_dir)
        path = visualiser.create_visualization(result=self.make_result(True))

        self.assertIsNotNone(path)
        self.assertTrue(Path(str(path)).is_file())

    def test_visualization_without_bbox(self) -> None:
        visualiser = Visualiser(show=False, save_path=self.temp_dir)
        path = visualiser.create_visualization(result=self.make_result(False))

        self.assertTrue(Path(str(path)).is_file())


if __name__ == "__main__":
    unittest.main()
