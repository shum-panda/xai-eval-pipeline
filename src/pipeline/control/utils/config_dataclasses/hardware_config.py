from dataclasses import dataclass


@dataclass
class HardwareConfig:
    use_cuda: bool = True
    device: str = "cuda:0"  # currently ignored, system checks availability
