# Usage example
import torch
from torch import nn

from xai_methods.explainer_registry import ExplainerRegistry
from xai_methods.xai_factory import XAIFactory

if __name__ == "__main__":
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )

    # Create factory and explainers
    factory = XAIFactory()

    print("Available explainers:", factory.list_available_explainers())

    # Create different explainers
    gradcam = factory.create_explainer("gradcam", model, target_layer="conv1")

    # Mock input
    images = torch.rand(2, 3, 224, 224)

    # Generate explanations
    gradcam_result = gradcam.explain(images)

    print(f"Grad-CAM output shape: {gradcam_result[0].shape}")
    # Verify singleton behavior
    registry1 = ExplainerRegistry.get_instance()
    registry2 = ExplainerRegistry()
    print(f"Same registry instance: {registry1 is registry2}")