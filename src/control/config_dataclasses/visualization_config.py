from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    save: bool = True
    show: bool = False
    overlay_heatmap: bool = True
    colormap: str = "jet"
    save_format: str = "png"
    dpi: int = 150
    max_visualizations: int = 50
