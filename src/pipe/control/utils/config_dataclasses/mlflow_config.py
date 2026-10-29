from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "xai_evaluation"
    run_name: Optional[str] = None
    auto_log: bool = True
    artifact_location: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
