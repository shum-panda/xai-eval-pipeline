from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "xai_evaluation"
    auto_log: bool = True
    artifact_location: Optional[str] = None
    tags: Optional[dict] = None