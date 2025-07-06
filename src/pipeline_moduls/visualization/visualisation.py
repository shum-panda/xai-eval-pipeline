import logging
from pathlib import Path
from typing import Optional

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from pipeline_moduls.evaluation.metric_calculator import bbox_to_mask_tensor
from pipeline_moduls.evaluation.dataclass.xai_metrics import XAIMetrics
from control.dataclasses.xai_explanation_result import XAIExplanationResult


class Visualiser:
    def __init__(self, show: bool = True, save_path: Optional[Path] = None):
        """
        Visualiser zur Darstellung von XAI-Ergebnissen.

        Args:
            show: Ob die Visualisierung angezeigt werden soll.
            save_path: Optionaler Pfad zum Speichern des Plots.
        """
        self.show = show
        self.save_path = save_path
        self.logger = logging.getLogger(__name__)

        # Seaborn-Stil aktivieren
        sns.set_theme(style="whitegrid")

    def create_visualization(
        self,
        result: XAIExplanationResult,
        metrics: Optional[XAIMetrics] = None
    ):
        """
        Erstelle Visualisierung eines XAI-Ergebnisses.

        Args:
            result: XAI Explanation Result
            metrics: Evaluation Metriken (optional)
        """
        try:
            # Lade Originalbild
            img_pil = Image.open(result.image_path)
            original_image = img_pil.resize((224, 224))

            # Setup Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original Image
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')

            # Attribution Heatmap mit Seaborn-Farbschema
            attribution = result.attribution.numpy().squeeze(0)
            sns.heatmap(
                attribution,
                ax=axes[1],
                cmap='rocket',  # Seaborn-CMAP, optional: 'magma', 'coolwarm', ...
                cbar=True,
                xticklabels=False,
                yticklabels=False,
                square=True
            )
            axes[1].set_title(f'{result.explainer_name} Attribution', fontsize=12)

            # Overlay: Originalbild + Attribution mit Transparenz
            axes[2].imshow(original_image)
            axes[2].imshow(attribution, cmap='rocket', alpha=0.4)

            # Bounding Box
            if result.has_bbox:
                bbox_mask = bbox_to_mask_tensor(result.bbox)
                if bbox_mask is not None:
                    axes[2].contour(
                        bbox_mask.squeeze().numpy(),
                        levels=[0.5],
                        colors=sns.color_palette("husl", 1),  # Seaborn-Farbe
                        linewidths=2
                    )

            axes[2].set_title('Attribution + BBox Overlay', fontsize=12)
            axes[2].axis('off')

            # Info-Text
            info_text = (
                f"Image: {result.image_name}\n"
                f"Model: {result.model_name}\n"
                f"Predicted: {result.predicted_class} "
                f"True: {result.true_label}\n"
                f"Correct: {'Yes' if result.prediction_correct else 'No'}\n"
            )

            if metrics:
                info_text += (
                    f"\nXAI Metrics:\n"
                    f"Pointing Game: {'Yes' if metrics.pointing_game_hit else 'No'}\n"
                    f"IoU: {metrics.iou_score:.3f}\n"
                    f"Coverage: {metrics.coverage_score:.3f}\n"
                    f"Precision: {metrics.precision:.3f}"
                )

            fig.suptitle(info_text, fontsize=10, ha='left')
            plt.tight_layout()

            if self.save_path:
                plt.savefig(self.save_path, bbox_inches='tight', dpi=150)
                self.logger.info(f"Visualisierung gespeichert: {self.save_path}")

            if self.show:
                plt.show()
            else:
                plt.close()

        except ImportError:
            self.logger.warning("Matplotlib oder Seaborn nicht verfügbar – keine Visualisierung")
