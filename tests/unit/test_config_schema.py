import shutil
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException

from src.pipe.control.utils.config_dataclasses.config_schema import (
    UnsupportedConfigError,
    check_supported,
    register_config_schema,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "experiments"


class TestConfigSchema(unittest.TestCase):

    def setUp(self) -> None:
        GlobalHydra.instance().clear()
        register_config_schema()
        self._hydra = initialize_config_dir(
            config_dir=str(CONFIG_DIR), version_base=None
        )
        self._hydra.__enter__()

    def tearDown(self) -> None:
        self._hydra.__exit__(None, None, None)
        GlobalHydra.instance().clear()

    def test_all_experiment_configs_match_schema(self) -> None:
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            with self.subTest(config=path.name):
                cfg = compose(config_name=path.stem)
                check_supported(cfg)

    def test_typo_in_yaml_key_fails(self) -> None:
        typo_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, typo_dir)
        (typo_dir / "typo.yaml").write_text(
            "defaults:\n  - base_config\n  - _self_\n\ndata:\n  batchsize: 8\n"
        )
        self._hydra.__exit__(None, None, None)
        GlobalHydra.instance().clear()
        self._hydra = initialize_config_dir(config_dir=str(typo_dir), version_base=None)
        self._hydra.__enter__()

        with self.assertRaises(ConfigCompositionException) as error:
            compose(config_name="typo")
        self.assertIn("batchsize", str(error.exception))

    def test_wrong_type_fails(self) -> None:
        with self.assertRaises(ConfigCompositionException):
            compose(config_name="config_test_quick", overrides=["data.batch_size=abc"])

    def test_unsupported_value_fails(self) -> None:
        cfg = compose(
            config_name="config_test_quick",
            overrides=[
                "model.weights_path=my.pt",
                "data.augmentation.random_crop=true",
            ],
        )
        with self.assertRaises(UnsupportedConfigError) as error:
            check_supported(cfg)
        self.assertIn("model.weights_path", str(error.exception))
        self.assertIn("data.augmentation.random_crop", str(error.exception))


if __name__ == "__main__":
    unittest.main()
