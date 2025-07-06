from dataclasses import dataclass, field


@dataclass
class XAIConfig:
    name: str = "gradcam"
    layer: str = "layer4"
    alpha: float = 0.5
    use_cuda: bool = True
    kwargs: dict = field(default_factory=lambda: {"guided_backprop": False})
