from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str = "my_xai_experiment"
    output_dir: str = "results/my_experiment"
    emergency_export_enabled: bool = True
    # seed: int = 42
    top_k: int = 10
