from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MetricResults:
    """
    Dynamische XAI-Metriken, gespeichert als Name → Ergebnis
    """

    values: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.values.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return self.values
