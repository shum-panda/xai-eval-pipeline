import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

from src.control.orchestrator import Orchestrator
from src.control.utils.config_dataclasses.master_config import MasterConfig


def run_multiple_configs(config_names: list[str]) -> None:
    with initialize(config_path="config"):
        for config_name in config_names:
            print(f"\n🔁 Running config: {config_name}")
            cfg = compose(config_name=config_name)
            print(OmegaConf.to_yaml(cfg))

            pipeline = Orchestrator(cfg)  # Deine Pipeline-Instanz
            pipeline.run()

            print(f"✅ Finished run for: {config_name}")


if __name__ == "__main__":
    run_multiple_configs([
        "config_grad_cam",
        "config_score_cam",
        "config_guided_backprop"
    ])