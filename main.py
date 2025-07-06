import hydra
from omegaconf import OmegaConf

from control.utils.config_dataclasses.master_config import MasterConfig
from control.xai_orchestrator import XAIOrchestrator

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: MasterConfig):
    """
    Main entry point for the XAI pipeline using Hydra configuration.
    """

    # 1) Print loaded config (optional, but very helpful)
    print("📄 Loaded Config:")
    print(OmegaConf.to_yaml(cfg))

    # 2) Initialize pipeline with Hydra config
    pipeline = XAIOrchestrator(cfg)

    # 3) Quick Test
    print("🧪 Running quick test...")
    test_result = pipeline.quick_test(3)
    print(f"✅ Quick test finished: {test_result['total_samples']} samples processed")

    # 4) Full Evaluation
    print("\n🚀 Running full evaluation...")
    result = pipeline.run()

    print("\n✅ Experiment completed!")
    print(f"📊 {result['total_samples']} samples processed")
    print(f"📁 Results saved in: {result['output_dir']}")

if __name__ == "__main__":
    main()
