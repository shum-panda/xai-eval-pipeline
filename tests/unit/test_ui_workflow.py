import unittest
from typing import Any, Dict, List, Tuple

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.pipe.control.utils.config_dataclasses.config_schema import (
    register_config_schema,
)
from src.ui.catalog import build_catalog
from src.ui.workflow import Workflow, compile_workflow

CATALOG = build_catalog()


def defaults(kind: str, type_: str) -> Dict[str, Any]:
    node = next(n for n in CATALOG["nodes"] if n["kind"] == kind and n["type"] == type_)
    return {p["key"]: p["default"] for p in node["params"]}


def graph(
    models: List[str],
    methods: List[str],
    metrics: Tuple[str, ...] = ("iou", "point_game"),
    name: str = "test flow",
) -> Dict[str, Any]:
    """dataset -> every model -> every method -> metrics (per method) -> experiment"""
    nodes = [
        {
            "id": "data",
            "kind": "dataset",
            "type": "imagenet_val",
            "params": defaults("dataset", "imagenet_val"),
        },
        {
            "id": "exp",
            "kind": "experiment",
            "type": "experiment",
            "params": defaults("experiment", "experiment"),
        },
    ]
    edges = []
    for model in models:
        nodes.append(
            {
                "id": model,
                "kind": "model",
                "type": model,
                "params": defaults("model", model),
            }
        )
        edges.append({"source": "data", "target": model})
    for method in methods:
        nodes.append(
            {
                "id": method,
                "kind": "xai",
                "type": method,
                "params": defaults("xai", method),
            }
        )
        for model in models:
            edges.append({"source": model, "target": method})
        for metric in metrics:
            metric_id = f"{method}-{metric}"
            nodes.append(
                {
                    "id": metric_id,
                    "kind": "metric",
                    "type": metric,
                    "params": defaults("metric", metric),
                }
            )
            edges.append({"source": method, "target": metric_id})
            edges.append({"source": metric_id, "target": "exp"})
    return {"name": name, "nodes": nodes, "edges": edges}


class TestCatalog(unittest.TestCase):

    def test_catalog_lists_registered_building_blocks(self) -> None:
        types = {(n["kind"], n["type"]) for n in CATALOG["nodes"]}
        for expected in [
            ("dataset", "imagenet_val"),
            ("model", "resnet50"),
            ("model", "vgg16"),
            ("xai", "grad_cam"),
            ("xai", "integrated_gradients"),
            ("metric", "iou"),
            ("metric", "point_game"),
            ("experiment", "experiment"),
        ]:
            self.assertIn(expected, types)

    def test_parameters_come_from_config_classes(self) -> None:
        grad_cam = defaults("xai", "grad_cam")
        self.assertEqual(grad_cam["interpolate_mode"], "bilinear")
        self.assertEqual(defaults("metric", "iou")["normalization"], "minmax")
        self.assertEqual(defaults("dataset", "imagenet_val")["data.resize"], [224, 224])


class TestCompileWorkflow(unittest.TestCase):

    def test_several_models_on_one_method_give_one_run_per_model(self) -> None:
        result = compile_workflow(
            Workflow.from_dict(graph(["resnet18", "resnet50"], ["grad_cam"]))
        )

        self.assertEqual(result.errors, [])
        self.assertEqual(
            [r.name for r in result.runs],
            ["test_flow_resnet18_grad_cam", "test_flow_resnet50_grad_cam"],
        )

    def test_several_methods_on_one_model(self) -> None:
        result = compile_workflow(
            Workflow.from_dict(graph(["resnet50"], ["grad_cam", "guided_backprop"]))
        )
        self.assertEqual(result.errors, [])
        self.assertEqual({r.xai for r in result.runs}, {"grad_cam", "guided_backprop"})

    def test_run_config_carries_graph_settings(self) -> None:
        flow = graph(["vgg16"], ["grad_cam"], metrics=("iou",))
        next(n for n in flow["nodes"] if n["id"] == "data")["params"][
            "data.batch_size"
        ] = 8
        next(n for n in flow["nodes"] if n["id"] == "grad_cam")["params"][
            "target_layer"
        ] = "-2"

        config = compile_workflow(Workflow.from_dict(flow)).runs[0].config

        self.assertEqual(config["defaults"], ["base_config", "_self_"])
        self.assertEqual(config["model"]["name"], "vgg16")
        self.assertEqual(config["data"]["batch_size"], 8)
        self.assertEqual(config["xai"]["kwargs"]["target_layer"], -2)
        self.assertEqual(config["metric"]["names"], ["iou"])
        self.assertEqual(config["metric"]["kwargs"]["iou"]["normalization"], "minmax")
        self.assertEqual(
            config["experiment"]["output_dir"], "results/test_flow/vgg16_grad_cam"
        )

    def test_missing_nodes_are_reported(self) -> None:
        result = compile_workflow(
            Workflow.from_dict({"name": "x", "nodes": [], "edges": []})
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("dataset" in e for e in result.errors))

    def test_method_without_metric_is_reported(self) -> None:
        flow = graph(["resnet50"], ["grad_cam"])
        flow["nodes"].append(
            {"id": "gb", "kind": "xai", "type": "guided_backprop", "params": {}}
        )
        flow["edges"].append({"source": "resnet50", "target": "gb"})

        result = compile_workflow(Workflow.from_dict(flow))

        self.assertIn("XAI method 'guided_backprop' has no metric.", result.errors)

    def test_edge_skipping_a_stage_is_rejected(self) -> None:
        flow = graph(["resnet50"], ["grad_cam"])
        flow["edges"].append({"source": "data", "target": "grad_cam"})
        result = compile_workflow(Workflow.from_dict(flow))
        self.assertIn(
            "A dataset node cannot be connected to a xai node.", result.errors
        )

    def test_invalid_value_is_caught_by_the_schema(self) -> None:
        flow = graph(["resnet50"], ["grad_cam"])
        next(n for n in flow["nodes"] if n["id"] == "data")["params"][
            "data.batch_size"
        ] = "many"
        result = compile_workflow(Workflow.from_dict(flow))
        self.assertTrue(any("batch_size" in e for e in result.errors))


class TestGeneratedYamlRunsThroughHydra(unittest.TestCase):
    """The generated YAML is an ordinary experiment config for main.py."""

    def test_yaml_composes_with_hydra(self) -> None:
        import tempfile
        from pathlib import Path

        run = compile_workflow(
            Workflow.from_dict(graph(["resnet50"], ["integrated_gradients"]))
        ).runs[0]
        with tempfile.TemporaryDirectory() as config_dir:
            (Path(config_dir) / f"{run.name}.yaml").write_text(run.yaml)
            GlobalHydra.instance().clear()
            register_config_schema()
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=run.name)
            GlobalHydra.instance().clear()

        self.assertEqual(cfg.xai.name, "integrated_gradients")
        self.assertEqual(cfg.xai.kwargs.n_steps, 50)
        self.assertEqual(list(cfg.metric.names), ["iou", "point_game"])


if __name__ == "__main__":
    unittest.main()
