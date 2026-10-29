import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchvision  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def resolve_project_path(path: str) -> Path:
    """
    Resolve a path from the config against the project root.

    Hydra may change the working directory, so relative paths in the YAML are
    always interpreted relative to the repository root, not the current directory.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def set_seeds(seed: int, deterministic: bool) -> None:
    """
    Seed all random number generators used by the pipeline.

    Args:
        seed: Seed for Python, NumPy and PyTorch (CPU and CUDA).
        deterministic: Additionally enforce deterministic PyTorch algorithms.
            Operations without a deterministic implementation only warn.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by cuBLAS for deterministic results on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_provenance() -> Dict[str, str]:
    """
    Collect the information needed to reproduce a run besides the config.

    Returns:
        Dict[str, str]: Git commit and state, library versions and platform.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": str(bool(status)) if commit else "unknown",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": str(torch.version.cuda),
        "platform": platform.platform(),
    }
