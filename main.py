import hydra
from omegaconf import OmegaConf

from pipeline.control.orchestrator import Orchestrator
from pipeline.control.utils import MasterConfig


@hydra.main(
    version_base=None,
    config_path="config/experiments",
    config_name="config_resnet50_integrated_gradients",
)
def main(cfg: MasterConfig) -> None:
    """
    Main entry point for the XAI pipeline using Hydra configuration.
    """

    # 1) Print loaded config (optional, but very helpful)
    print("Loaded Config:")
    print(OmegaConf.to_yaml(cfg))

    # 2) Initialize pipeline with Hydra config
    pipeline = Orchestrator(cfg)

    # 3) Full Evaluation
    print("\nRunning full evaluation...")
    pipeline.run()

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
