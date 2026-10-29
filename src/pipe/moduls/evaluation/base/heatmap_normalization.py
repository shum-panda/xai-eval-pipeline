from typing import Any, Dict

import torch

HEATMAP_NORMALIZATIONS = ("none", "minmax", "max")


def get_normalization(kwargs: Dict[str, Any]) -> str:
    """
    Read and validate the 'normalization' option from metric kwargs.

    Args:
        kwargs: Metric kwargs from the YAML config.

    Returns:
        str: One of HEATMAP_NORMALIZATIONS. Defaults to "none".

    Raises:
        ValueError: If the value is not a known normalization.
    """
    normalization = kwargs.get("normalization", "none")
    if normalization not in HEATMAP_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            f"Expected one of {HEATMAP_NORMALIZATIONS}."
        )
    return str(normalization)


def normalize_heatmap(heatmap: torch.Tensor, normalization: str) -> torch.Tensor:
    """
    Scale a heatmap before it is binarized with a fixed threshold.

    XAI methods return attributions on different, model-dependent scales.
    Without normalization a fixed threshold compares magnitudes instead of
    localization, which makes results between methods and models incomparable.

    Args:
        heatmap: Attribution map of shape [H, W].
        normalization:
            - "none": use raw values (reproduces runs before normalization existed)
            - "minmax": scale to [0, 1] via (h - min) / (max - min)
            - "max": divide by the maximum absolute value

    Returns:
        torch.Tensor: The scaled heatmap. Constant maps (all values equal) are
        returned as zeros for "minmax" and unchanged otherwise.
    """
    if normalization == "none":
        return heatmap

    heatmap = heatmap.float()
    if normalization == "minmax":
        span = heatmap.max() - heatmap.min()
        if span == 0:
            return torch.zeros_like(heatmap)
        return (heatmap - heatmap.min()) / span

    if normalization == "max":
        peak = heatmap.abs().max()
        if peak == 0:
            return heatmap
        return heatmap / peak

    raise ValueError(f"Unknown normalization '{normalization}'.")
