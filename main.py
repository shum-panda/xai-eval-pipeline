import hydra
from omegaconf import OmegaConf

from src.control.utils.config_dataclasses.master_config import MasterConfig
from src.control.xai_orchestrator import XAIOrchestrator


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: MasterConfig) -> None:
    """
    Main entry point for the XAI pipeline using Hydra configuration.
    """

    # 1) Print loaded config (optional, but very helpful)
    print("📄 Loaded Config:")
    print(OmegaConf.to_yaml(cfg))

    # 2) Initialize pipeline with Hydra config
    pipeline = XAIOrchestrator(cfg)

    # # 3) Quick Test
    # print("🧪 Running quick test...")
    # test_result = pipeline.quick_test(3)
    # print(f"✅ Quick test finished: {test_result['total_results']} samples processed")

    # 4) Full Evaluation
    print("\nRunning full evaluation...")
    pipeline.run()

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
