from dataclasses import dataclass, field

from src.pipeline.control.utils.config_dataclasses.data_config import DataConfig
from src.pipeline.control.utils.config_dataclasses.experiment_config import (
    ExperimentConfig,
)
from src.pipeline.control.utils.config_dataclasses.hardware_config import HardwareConfig
from src.pipeline.control.utils.config_dataclasses.logging_config import LoggingConfig
from src.pipeline.control.utils.config_dataclasses.metric_config import MetricConfig
from src.pipeline.control.utils.config_dataclasses.mlflow_config import MLflowConfig
from src.pipeline.control.utils.config_dataclasses.model_config import ModelConfig
from src.pipeline.control.utils.config_dataclasses.visualization_config import (
    VisualizationConfig,
)
from src.pipeline.control.utils.config_dataclasses.xai_config import XAIConfig


@dataclass
class MasterConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
