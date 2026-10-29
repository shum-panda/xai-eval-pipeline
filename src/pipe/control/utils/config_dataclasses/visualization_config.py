from dataclasses import dataclass
from typing import Optional


@dataclass
class VisualizationConfig:
    save: bool = True
    show: bool = False
    max_visualizations: Optional[int] = 50  # null saves all
    # Note: colormap, overlay_heatmap, save_format, dpi are hardcoded in
    # visualisation.py
