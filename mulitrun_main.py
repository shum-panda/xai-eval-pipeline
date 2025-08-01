import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

from src.control.orchestrator import Orchestrator
from src.control.utils.config_dataclasses.master_config import MasterConfig


def run_multiple_configs(config_names: list[str]) -> None:
    with initialize(version_base=None, config_path="config/experiments"):
        for config_name in config_names:
            try:
                print(f"\n🔁 Running config: {config_name}")
                cfg = compose(config_name=config_name)
                print(OmegaConf.to_yaml(cfg))

                pipeline = Orchestrator(cfg)  # Deine Pipeline-Instanz
                pipeline.run()

                print(f"✅ Finished run for: {config_name}")
            except Exception as e:
                print(f"while {config_name} a error accorded: {e}")


if __name__ == "__main__":
    run_multiple_configs([
        "config_score_cam",
        "config_guided_backprop",
        "config_integrated_gradient",
        "config_grad_cam_vgg16",
    ])