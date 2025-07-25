from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MetricConfig:
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})
