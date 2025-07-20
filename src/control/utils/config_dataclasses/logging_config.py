from dataclasses import dataclass


@dataclass
class LoggingConfig:
    level: str = "INFO"
