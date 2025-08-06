from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    save: bool = True
    show: bool = False
    max_visualizations: int = 50
    # Note: colormap, overlay_heatmap, save_format, dpi are hardcoded in visualisation.py
