from dataclasses import dataclass

from control.utils.config_dataclasses.data_config import DataConfig
from control.utils.config_dataclasses.experiment_config import ExperimentConfig
from control.utils.config_dataclasses.hardware_config import HardwareConfig
from control.utils.config_dataclasses.logging_config import LoggingConfig
from control.utils.config_dataclasses.metrics_config import MetricsConfig
from control.utils.config_dataclasses.model_config import ModelConfig
from control.utils.config_dataclasses.reporting_config import ReportingConfig
from control.utils.config_dataclasses.visualization_config import VisualizationConfig
from control.utils.config_dataclasses.xai_config import XAIConfig


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
