import hydra
from omegaconf import OmegaConf

from src.pipe.control.orchestrator import Orchestrator
from src.pipe.control.utils.config_dataclasses.config_schema import (
    check_supported,
    register_config_schema,
)
from src.pipe.control.utils.config_dataclasses.master_config import MasterConfig

# Must happen before @hydra.main composes the config
register_config_schema()


@hydra.main(
    version_base=None,
    config_path="config/experiments",
    config_name="config_test_quick",
)
def main(cfg: MasterConfig) -> None:
    """
    Main entry point for the XAI pipeline using Hydra configuration.

    Select an experiment with --config-name, e.g.
    python main.py --config-name config_resnet50_grad_cam
    """

    # 1) The schema already rejected unknown keys and wrong types; reject
    #    values the pipeline cannot honour yet
    check_supported(cfg)

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
