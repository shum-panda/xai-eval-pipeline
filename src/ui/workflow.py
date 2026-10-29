"""
Compile a workflow graph from the UI into pipeline configs.

The graph is only an editor for configs: every run it describes becomes an
ordinary experiment YAML that is validated against the same schema as a
hand-written one and executed by main.py. The YAML stays the single source of
truth.

Graph shape (edges only go from one kind to the next):

    dataset -> model(s) -> XAI method(s) -> metric(s) -> experiment

Each connected (model, XAI method) pair becomes one run. Several models on one
method, or several methods on one model, give the comparisons the pipeline is
built for.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from src.pipe.control.utils.config_dataclasses.config_schema import (
    SCHEMA_NAME,
    check_supported,
)
from src.pipe.control.utils.config_dataclasses.master_config import MasterConfig
from src.ui.catalog import KIND_ORDER


@dataclass
class Node:
    id: str
    kind: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str


@dataclass
class Workflow:
    name: str
    nodes: List[Node]
    edges: List[Edge]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        return cls(
            name=str(data.get("name") or "workflow"),
            nodes=[
                Node(
                    id=str(n["id"]),
                    kind=str(n["kind"]),
                    type=str(n["type"]),
                    params=dict(n.get("params") or {}),
                )
                for n in data.get("nodes", [])
            ],
            edges=[
                Edge(source=str(e["source"]), target=str(e["target"]))
                for e in data.get("edges", [])
            ],
        )


@dataclass
class RunConfig:
    name: str
    model: str
    xai: str
    config: Dict[str, Any]
    yaml: str


@dataclass
class CompileResult:
    runs: List[RunConfig] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.runs)


def slug(text: str) -> str:
    """File-system and MLflow friendly name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return cleaned or "workflow"


def _nest(flat: Dict[str, Any]) -> Dict[str, Any]:
    """{"data.batch_size": 8} -> {"data": {"batch_size": 8}}"""
    nested: Dict[str, Any] = {}
    for key, value in flat.items():
        section, _, name = key.partition(".")
        nested.setdefault(section, {})[name] = value
    return nested


def _layer(value: Any) -> Any:
    """Target layers are an index (also typed as text in the UI) or a name."""
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return value


def _validate(config: Dict[str, Any]) -> Optional[str]:
    """Validate like Hydra does for a YAML that uses the base_config schema."""
    try:
        body = {k: v for k, v in config.items() if k != "defaults"}
        merged = OmegaConf.merge(OmegaConf.structured(MasterConfig), body)
        check_supported(merged)  # type: ignore[arg-type]
    except Exception as error:  # noqa: BLE001 - report every config problem
        return str(error)
    return None


def compile_workflow(workflow: Workflow) -> CompileResult:
    """
    Turn the graph into one validated config per (model, XAI method) pair.

    Returns:
        CompileResult: The runs, or a list of errors explaining what is missing.
    """
    result = CompileResult()
    nodes = {node.id: node for node in workflow.nodes}

    for kind in KIND_ORDER:
        if kind in ("dataset", "experiment"):
            count = sum(1 for n in workflow.nodes if n.kind == kind)
            if count != 1:
                result.errors.append(
                    f"The workflow needs exactly one {kind} node (found {count})."
                )
        elif not any(n.kind == kind for n in workflow.nodes):
            result.errors.append(f"Add at least one {kind} node.")

    outgoing: Dict[str, List[Node]] = {node_id: [] for node_id in nodes}
    incoming: Dict[str, List[Node]] = {node_id: [] for node_id in nodes}
    for edge in workflow.edges:
        source, target = nodes.get(edge.source), nodes.get(edge.target)
        if source is None or target is None:
            result.errors.append("An edge points to a node that does not exist.")
            continue
        if source.kind not in KIND_ORDER or target.kind not in KIND_ORDER:
            result.errors.append(f"Unknown node kind on edge {source.id}->{target.id}.")
            continue
        if KIND_ORDER.index(target.kind) != KIND_ORDER.index(source.kind) + 1:
            result.errors.append(
                f"A {source.kind} node cannot be connected to a {target.kind} node."
            )
            continue
        outgoing[source.id].append(target)
        incoming[target.id].append(source)

    if result.errors:
        return result

    dataset = next(n for n in workflow.nodes if n.kind == "dataset")
    experiment = next(n for n in workflow.nodes if n.kind == "experiment")
    workflow_slug = slug(workflow.name)

    for model in (n for n in workflow.nodes if n.kind == "model"):
        if dataset not in incoming[model.id]:
            result.errors.append(f"Model '{model.type}' is not connected to the data.")
            continue
        methods = outgoing[model.id]
        if not methods:
            result.errors.append(f"Model '{model.type}' has no XAI method.")
        for xai in methods:
            metrics = outgoing[xai.id]
            if not metrics:
                result.errors.append(f"XAI method '{xai.type}' has no metric.")
                continue
            metric_types = [m.type for m in metrics]
            if len(set(metric_types)) != len(metric_types):
                result.errors.append(
                    f"XAI method '{xai.type}' has the same metric connected twice."
                )
                continue
            unconnected = [m.type for m in metrics if experiment not in outgoing[m.id]]
            if unconnected:
                result.errors.append(
                    f"Metric(s) {', '.join(unconnected)} not connected "
                    "to the experiment."
                )
                continue
            result.runs.append(
                _build_run(workflow_slug, dataset, model, xai, metrics, experiment)
            )

    names = [run.name for run in result.runs]
    for name in sorted({n for n in names if names.count(n) > 1}):
        result.errors.append(
            f"Run '{name}' is defined twice (same model and method connected twice)."
        )

    for run in result.runs:
        error = _validate(run.config)
        if error:
            result.errors.append(f"{run.name}: {error}")

    if not result.runs and not result.errors:
        result.errors.append("The workflow does not describe any run.")
    return result


def _build_run(
    workflow_slug: str,
    dataset: Node,
    model: Node,
    xai: Node,
    metrics: List[Node],
    experiment: Node,
) -> RunConfig:
    name = f"{workflow_slug}_{model.type}_{xai.type}"
    settings = _nest(experiment.params)
    xai_kwargs = {
        key: _layer(value) if key == "target_layer" else value
        for key, value in xai.params.items()
    }

    config: Dict[str, Any] = {
        "defaults": [SCHEMA_NAME, "_self_"],
        "experiment": {
            **settings.get("experiment", {}),
            "name": name,
            "output_dir": f"results/{workflow_slug}/{model.type}_{xai.type}",
        },
        "model": {
            "name": model.type,
            "pretrained": bool(model.params.get("pretrained", True)),
        },
        "xai": {"name": xai.type, "use_defaults": True, "kwargs": xai_kwargs},
        "metric": {
            "names": [m.type for m in metrics],
            "kwargs": {m.type: dict(m.params) for m in metrics if m.params},
        },
        "data": _nest(dataset.params).get("data", {}),
        "hardware": settings.get("hardware", {}),
        "visualization": settings.get("visualization", {}),
        "logging": settings.get("logging", {}),
        "mlflow": {
            **settings.get("mlflow", {}),
            "run_name": name,
            "tags": {
                "workflow": workflow_slug,
                "model": model.type,
                "xai_method": xai.type,
                "created_with": "pipeline_ui",
            },
        },
    }
    yaml = (
        f"# Generated by the pipeline UI from workflow '{workflow_slug}'.\n"
        "# Run directly with: python main.py --config-dir "
        f"config/generated/{workflow_slug} --config-name {name}\n"
        + OmegaConf.to_yaml(OmegaConf.create(config))
    )
    return RunConfig(
        name=name, model=model.type, xai=xai.type, config=config, yaml=yaml
    )
