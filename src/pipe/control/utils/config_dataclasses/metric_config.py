from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricConfig:
    # Registered metric names to compute; null computes all built-in metrics
    names: Optional[List[str]] = None
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})
