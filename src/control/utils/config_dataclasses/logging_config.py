from dataclasses import dataclass


@dataclass
class LoggingConfig:
    level: str = "INFO"
    to_file: bool = True
    file: str = "logs/experiment.log"
