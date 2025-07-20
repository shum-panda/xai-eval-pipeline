from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricsConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "explanation_faithfulness"]
    )
