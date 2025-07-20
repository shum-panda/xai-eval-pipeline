import logging

from control.utils.config_dataclasses.logging_config import LoggingConfig


def setup_logger(logging_config: LoggingConfig):
    """
    Sets the global logging level based on the provided LoggingConfig.

    This function configures the root logger to use the log level defined
    in the LoggingConfig dataclass. It does not configure handlers, formats,
    or file output – Hydra takes care of those aspects automatically.

    Args:
        logging_config (LoggingConfig): The logging configuration object,
        typically loaded from a Hydra YAML config.

    Raises:
        ValueError: If the provided log level is not a valid logging level.
    """
    numeric_level = getattr(logging, logging_config.level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {logging_config.level}")

    logging.getLogger().setLevel(numeric_level)