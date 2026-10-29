from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.pipe.control.utils.config_dataclasses.master_config import MasterConfig

#: Name under which MasterConfig is stored as schema. Experiment YAMLs reference
#: it in their defaults list, so Hydra validates every key and type against it.
SCHEMA_NAME = "base_config"


class UnsupportedConfigError(ValueError):
    """Raised when a config value is valid YAML but not supported by the pipeline."""


def register_config_schema() -> None:
    """
    Register MasterConfig as structured config schema in Hydra's ConfigStore.

    With the schema, unknown keys (e.g. typos like 'batchsize') and wrong types
    fail before the run starts instead of being silently ignored.
    """
    ConfigStore.instance().store(name=SCHEMA_NAME, node=MasterConfig)


def check_supported(cfg: DictConfig) -> None:
    """
    Fail early on options the schema accepts but the pipeline cannot honour yet.

    The config is the single source of truth: an option that is logged to MLflow
    must also take effect. Options that are not implemented therefore raise an
    error instead of being ignored.

    Raises:
        UnsupportedConfigError: Listing every unsupported value found.
    """
    problems: List[str] = []

    def unsupported(path: str, value: Any, reason: str) -> None:
        problems.append(f"{path}={value!r}: {reason}")

    if cfg.model.weights_path is not None:
        unsupported(
            "model.weights_path",
            cfg.model.weights_path,
            "loading custom weights is not implemented yet",
        )
    if not cfg.data.normalize:
        unsupported(
            "data.normalize",
            cfg.data.normalize,
            "the models expect ImageNet-normalized input",
        )
    for name, enabled in cfg.data.augmentation.items():
        if enabled:
            unsupported(
                f"data.augmentation.{name}",
                enabled,
                "augmentation is not applied during evaluation",
            )
    if len(cfg.data.resize) != 2:
        unsupported("data.resize", list(cfg.data.resize), "needs exactly two values")

    if problems:
        raise UnsupportedConfigError(
            "Unsupported config values:\n  - " + "\n  - ".join(problems)
        )
