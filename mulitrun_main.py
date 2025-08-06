import gc
import traceback

import torch
from hydra import initialize, compose

from src.pipeline.control.orchestrator import Orchestrator


def cleanup_singletons():
    """Force cleanup of singleton instances between runs"""
    # Clear model factory singleton state
    from src.pipeline.pipeline_moduls.models.model_registry import ModelRegistry
    if ModelRegistry._instance is not None:
        ModelRegistry._instance._current_model = None
    
    # Clear explainer registry singleton state  
    from src.pipeline.pipeline_moduls.xai_methods.explainer_registry import (
        ExplainerRegistry)
    if ExplainerRegistry._instance is not None:
        # Don't clear the registry itself, just ensure fresh factory instances
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_multiple_configs(config_names: list[str]) -> None:
    with initialize(version_base=None, config_path="config/experiments"):
        for i, config_name in enumerate(config_names):
            try:
                print(f"\n🔁 Running config {i+1}/{len(config_names)}: {config_name}")
                
                # Clean up before each run (except the first)
                if i > 0:
                    cleanup_singletons()
                
                cfg = compose(config_name=config_name)
                print(f"   XAI Method: {cfg.xai.name}")
                print(f"   Model: {cfg.model.name}")
                print(f"   Output Dir: {cfg.experiment.output_dir}")

                # Create fresh orchestrator instance
                pipeline = Orchestrator(cfg)
                result = pipeline.run()
                
                # Explicit cleanup after each run
                pipeline._model_factory.unload_current_model()
                del pipeline
                
                print(f"✅ Finished run for: {config_name}")
                print(f"   Status: {result['status']}")
                print(f"   Samples: {result.get('total_samples', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error in {config_name}: {e}")
                traceback.print_exc()
                continue

    # Nach allen Einzelläufen: Erstelle Vergleichsplots
    print("\n📊 Creating comparison analysis...")
    try:
        from src.analyse.simple_analyzer import SimpleAnalyzer
        
        analyzer = SimpleAnalyzer()
        analyzer.run_all_analyses()
        
        print("✅ Comparison analysis completed!")
        
    except Exception as e:
        print(f"❌ Error creating comparison analysis: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_multiple_configs([
        "config_vgg16_grad_cam",
        "config_resnet18_grad_cam",
        "config_resnet34_grad_cam",
        "config_resnet50_guided_backprop",
        "config_resnet50_integrated_gradients",
    ])