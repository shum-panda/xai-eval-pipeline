from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.pipe.moduls.visualization.visualisation import Visualiser


def test_visualiser_layout():
    # Dummy-Bild (RGB, 224x224)
    dummy_image_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_image = Image.fromarray(dummy_image_array)
    dummy_image_path = Path("dummy_image.jpg")
    dummy_image.save(dummy_image_path)  # Temporär speichern

    # Dummy Attribution Map (shape: 1x14x14, wie in echten Beispielen)
    dummy_attribution = torch.rand(1, 14, 14)

    # Dummy bbox-Maske (optional)
    dummy_bbox = [60, 100, 160, 180]  # xmin, ymin, xmax, ymax

    # Fake XAIExplanationResult (SimpleNamespace als Ersatz)
    dummy_result = SimpleNamespace(
        image_path=dummy_image_path,
        image_name="DummyImage.jpg",
        model_name="DummyModel",
        explainer_name="DummyExplainer",
        predicted_class=123,
        predicted_class_name="test",
        true_label=456,
        true_label_name="test2",
        prediction_correct=False,
        attribution=dummy_attribution,
        has_bbox=True,
        bbox=dummy_bbox,
    )

    # Visualiser aufrufen
    save_path = Path("results/test")
    save_path.mkdir(parents=True, exist_ok=True)
    vis = Visualiser(show=True, save_path=save_path)
    vis.create_visualization(result=dummy_result)

    # Optional: Dummy-Datei löschen
    dummy_image_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_visualiser_layout()
