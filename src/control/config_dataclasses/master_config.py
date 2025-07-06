from dataclasses import dataclass

from control.config_dataclasses.data_config import DataConfig
from control.config_dataclasses.experiment_config import ExperimentConfig
from control.config_dataclasses.hardware_config import HardwareConfig
from control.config_dataclasses.logging_config import LoggingConfig
from control.config_dataclasses.metrics_config import MetricsConfig
from control.config_dataclasses.model_config import ModelConfig
from control.config_dataclasses.reporting_config import ReportingConfig
from control.config_dataclasses.visualization_config import VisualizationConfig
from control.config_dataclasses.xai_config import XAIConfig


@dataclass
class MasterConfig:
    experiment: ExperimentConfig = ExperimentConfig()
    hardware: HardwareConfig = HardwareConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    xai: XAIConfig = XAIConfig()
    metrics: MetricsConfig = MetricsConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    logging: LoggingConfig = LoggingConfig()
    reporting: ReportingConfig = ReportingConfig()
