from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class XAIConfig:
    name: str = "grad_cam"
    use_defaults: bool = True
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"guided_backprop": False})
