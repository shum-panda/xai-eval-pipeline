import hydra
from omegaconf import OmegaConf

from src.pipeline.control.utils.config_dataclasses.master_config import MasterConfig
from src.pipeline.control.orchestrator import Orchestrator



@hydra.main(
    version_base=None,
    config_path="config/experiments",
    config_name="config_test_quick",
)
def main(cfg: MasterConfig) -> None:
    """
    Main entry point for the XAI pipeline using Hydra configuration.
    """

    # 1) Disable struct mode to allow missing fields
    OmegaConf.set_struct(cfg, False)

    # 2) Print loaded config (optional, but very helpful)
    print("Loaded Config:")
    print(OmegaConf.to_yaml(cfg))

    # 3) Initialize pipeline with Hydra config
    pipeline = Orchestrator(cfg)

    # 3) Full Evaluation
    print("\nRunning full evaluation...")
    pipeline.run()

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
