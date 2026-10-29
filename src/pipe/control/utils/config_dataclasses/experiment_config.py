from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str = "my_xai_experiment"
    output_dir: str = "results/my_experiment"
    seed: int = 42
    top_k: int = 10
    # Enforce deterministic PyTorch algorithms (slower; ops without a
    # deterministic implementation only warn)
    deterministic: bool = False
