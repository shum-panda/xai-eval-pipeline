from dataclasses import dataclass


@dataclass
class HardwareConfig:
    use_cuda: bool = True
    device: str = "cuda:0"  # used when use_cuda is true and CUDA is available
