"""
Catalog of the building blocks offered in the pipeline UI.

Everything is read from the registries and config dataclasses, so a newly
registered model, XAI method or metric shows up in the UI without UI changes.
"""

import dataclasses
import typing
from typing import Any, Dict, List, Optional, Tuple, Type

import src.pipe.moduls.evaluation.metrics  # noqa: F401  (registers the metrics)
from src.pipe.control.utils.config_dataclasses.data_config import DataConfig
from src.pipe.control.utils.config_dataclasses.experiment_config import (
    ExperimentConfig,
)
from src.pipe.control.utils.config_dataclasses.hardware_config import HardwareConfig
from src.pipe.control.utils.config_dataclasses.logging_config import LoggingConfig
from src.pipe.control.utils.config_dataclasses.mlflow_config import MLflowConfig
from src.pipe.control.utils.config_dataclasses.visualization_config import (
    VisualizationConfig,
)
from src.pipe.moduls.evaluation.base.metric_registry import MetricRegistry
from src.pipe.moduls.models.xai_model_factory import XAIModelFactory
from src.pipe.moduls.xai_methods.xai_factory import XAIFactory

#: Node kinds in pipeline order. An edge may only go from one kind to the next.
KIND_ORDER = ["dataset", "model", "xai", "metric", "experiment"]

#: Fields of the dataset node: (config section, field name)
DATASET_FIELDS = [
    ("data", "dataset_path"),
    ("data", "annotation_path"),
    ("data", "label_file"),
    ("data", "resize"),
    ("data", "batch_size"),
    ("data", "max_batches"),
    ("data", "num_workers"),
    ("data", "shuffle"),
    ("data", "pin_memory"),
]

#: Fields of the experiment node: (config section, field name)
EXPERIMENT_FIELDS = [
    ("experiment", "seed"),
    ("experiment", "deterministic"),
    ("experiment", "top_k"),
    ("hardware", "use_cuda"),
    ("hardware", "device"),
    ("mlflow", "experiment_name"),
    ("mlflow", "tracking_uri"),
    ("visualization", "save"),
    ("visualization", "max_visualizations"),
    ("logging", "level"),
]

SECTION_CLASSES: Dict[str, Type[Any]] = {
    "data": DataConfig,
    "experiment": ExperimentConfig,
    "hardware": HardwareConfig,
    "mlflow": MLflowConfig,
    "visualization": VisualizationConfig,
    "logging": LoggingConfig,
}

FIELD_OPTIONS: Dict[str, List[Any]] = {
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR"],
}


def _type_name(hint: Any) -> Dict[str, Any]:
    """Map a Python type hint to the UI parameter type."""
    nullable = False
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        nullable = len(args) < len(typing.get_args(hint))
        hint = args[0] if len(args) == 1 else str
        origin = typing.get_origin(hint)
    if origin in (list, List):
        return {"type": "int_list", "nullable": nullable}
    names = {bool: "bool", int: "int", float: "float", str: "str"}
    return {"type": names.get(hint, "str"), "nullable": nullable}


def _param(
    key: str,
    default: Any,
    hint: Any = None,
    options: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    spec = _type_name(hint if hint is not None else type(default))
    if options:
        spec["type"] = "enum"
        spec["options"] = options
    spec.update({"key": key, "default": default})
    return spec


def _section_params(fields: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    params = []
    for section, name in fields:
        cls = SECTION_CLASSES[section]
        default = getattr(cls(), name)
        hint = typing.get_type_hints(cls)[name]
        key = f"{section}.{name}"
        params.append(_param(key, default, hint, FIELD_OPTIONS.get(key)))
    return params


def _xai_params(explainer_cls: Any) -> List[Dict[str, Any]]:
    config_cls = getattr(explainer_cls, "config_class", None)
    if config_cls is None:
        return []
    defaults = config_cls.get_defaults()
    options = config_cls.get_options()
    hints = typing.get_type_hints(config_cls)
    params = []
    for field in dataclasses.fields(config_cls):
        if field.name == "use_defaults" or field.name not in defaults:
            continue
        default = defaults[field.name]
        if field.name == "target_layer":
            # Index (-1 = last conv layer) or layer name
            params.append(
                {
                    "key": field.name,
                    "type": "layer",
                    "nullable": False,
                    "default": default,
                }
            )
            continue
        params.append(
            _param(field.name, default, hints.get(field.name), options.get(field.name))
        )
    return params


def _label(name: str) -> str:
    special = {
        "iou": "IoU",
        "vgg16": "VGG16",
        "grad_cam": "Grad-CAM",
        "score_cam": "Score-CAM",
    }
    if name in special:
        return special[name]
    if name.startswith("resnet"):
        return "ResNet" + name[len("resnet") :]
    return name.replace("_", " ").title()


def build_catalog() -> Dict[str, Any]:
    """
    Describe every node type the UI can place on the canvas.

    Returns:
        dict: {"kinds": [...], "nodes": [...]} where each node has kind, type,
        label and a list of parameter specs (key, type, default, ...).
    """
    nodes: List[Dict[str, Any]] = [
        {
            "kind": "dataset",
            "type": "imagenet_val",
            "label": "ImageNet Validation",
            "params": _section_params(DATASET_FIELDS),
        }
    ]

    for name in sorted(XAIModelFactory().list_available()):
        nodes.append(
            {
                "kind": "model",
                "type": name,
                "label": _label(name),
                "params": [_param("pretrained", True)],
            }
        )

    xai_factory = XAIFactory()
    for name in sorted(xai_factory.list_available_explainers()):
        nodes.append(
            {
                "kind": "xai",
                "type": name,
                "label": _label(name),
                "params": _xai_params(xai_factory.registry.get(name)),
            }
        )

    for name, metric_cls in sorted(MetricRegistry.list_metrics().items()):
        defaults = metric_cls.get_defaults()
        options = metric_cls.get_options()
        nodes.append(
            {
                "kind": "metric",
                "type": name,
                "label": _label(name),
                "params": [
                    _param(key, value, options=options.get(key))
                    for key, value in defaults.items()
                ],
            }
        )

    nodes.append(
        {
            "kind": "experiment",
            "type": "experiment",
            "label": "Experiment",
            "params": _section_params(EXPERIMENT_FIELDS),
        }
    )
    return {"kinds": KIND_ORDER, "nodes": nodes}
