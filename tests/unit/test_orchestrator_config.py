import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest import mock

import torch
from omegaconf import OmegaConf

from src.pipe.control.orchestrator import Orchestrator
from src.pipe.control.utils.config_dataclasses.master_config import MasterConfig
from src.pipe.control.utils.provenance import PROJECT_ROOT


def make_orchestrator(**overrides: Any) -> Orchestrator:
    """Orchestrator with config only, without loading a model or data."""
    config = OmegaConf.structured(MasterConfig)
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._config = config
    orchestrator._logger = mock.MagicMock()
    return orchestrator


class TestResolveDevice(unittest.TestCase):

    def test_cpu_when_cuda_is_disabled(self) -> None:
        orchestrator = make_orchestrator(**{"hardware.use_cuda": False})
        self.assertEqual(orchestrator._resolve_device(), torch.device("cpu"))

    def test_configured_device_when_cuda_is_available(self) -> None:
        orchestrator = make_orchestrator(**{"hardware.device": "cuda:1"})
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(orchestrator._resolve_device(), torch.device("cuda:1"))

    def test_falls_back_to_cpu_with_warning(self) -> None:
        orchestrator = make_orchestrator(**{"hardware.use_cuda": True})
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(orchestrator._resolve_device(), torch.device("cpu"))
        orchestrator._logger.warning.assert_called_once()


class TestSetupDataloaderUsesConfigPaths(unittest.TestCase):

    def _setup(self, orchestrator: Orchestrator, project_root: Any) -> Dict[str, Any]:
        fake_loader = mock.MagicMock()
        fake_loader.__len__.return_value = 1
        fake_loader.dataset = [0]
        with mock.patch(
            "src.pipe.control.orchestrator.create_dataloader", return_value=fake_loader
        ) as create:
            orchestrator.setup_dataloader(
                project_root=project_root,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                target_size=[224, 224],
                transform=None,
            )
        return create.call_args.kwargs

    def test_relative_paths_from_yaml_are_resolved_against_project_root(self) -> None:
        orchestrator = make_orchestrator(
            **{
                "data.dataset_path": "my/images",
                "data.annotation_path": "my/boxes",
                "data.label_file": "my/labels.txt",
            }
        )
        kwargs = self._setup(orchestrator, project_root=None)
        self.assertEqual(kwargs["image_dir"], PROJECT_ROOT / "my" / "images")
        self.assertEqual(kwargs["annot_dir"], PROJECT_ROOT / "my" / "boxes")
        self.assertEqual(kwargs["label_file"], PROJECT_ROOT / "my" / "labels.txt")

    def test_explicit_project_root_is_used_as_base(self) -> None:
        orchestrator = make_orchestrator(**{"data.dataset_path": "imgs"})
        kwargs = self._setup(orchestrator, project_root=Path("base"))
        self.assertEqual(kwargs["image_dir"], Path("base") / "imgs")


class TestLogProvenance(unittest.TestCase):

    def test_resolved_config_and_tags_are_logged(self) -> None:
        orchestrator = make_orchestrator(**{"experiment.seed": 7})
        orchestrator._model = SimpleNamespace(device=torch.device("cpu"))
        with mock.patch("src.pipe.control.orchestrator.mlflow") as mlflow:
            orchestrator._log_provenance()

        tags = mlflow.set_tags.call_args.args[0]
        self.assertEqual(tags["device"], "cpu")
        self.assertIn("git_commit", tags)
        config_text, artifact_path = mlflow.log_text.call_args.args
        self.assertEqual(artifact_path, "config/resolved_config.yaml")
        self.assertIn("seed: 7", config_text)


if __name__ == "__main__":
    unittest.main()
