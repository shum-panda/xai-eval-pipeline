import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from pipeline_moduls.evaluation.xai_evaluator import bbox_to_mask_tensor


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
        self, result: XAIExplanationResult, metrics
    ) -> Optional[str]:
        """
        Erstellt eine Visualisierung für ein XAI Ergebnis.

        Args:
            result: XAI Explanation Result
            metrics: Kann EvaluationSummary oder individuelles Metrics-Objekt sein
        """

        # Extrahiere Metrik-Werte abhängig vom Typ
        if hasattr(metrics, "metric_averages"):
            # EvaluationSummary - verwende Durchschnittswerte
            iou_score = metrics.metric_averages.get("average_IoU", 0.0)
            point_game_score = metrics.metric_averages.get("average_point_game", 0.0)
            pixel_precision = metrics.metric_averages.get(
                "average_PixelPrecisionRecall_precision", 0.0
            )
            pixel_recall = metrics.metric_averages.get(
                "average_PixelPrecisionRecall_recall", 0.0
            )
        elif hasattr(metrics, "iou_score"):
            # Individuelles Metrics-Objekt
            iou_score = getattr(metrics, "iou_score", 0.0)
            point_game_score = getattr(metrics, "point_game_score", 0.0)
            pixel_precision = getattr(metrics, "pixel_precision", 0.0)
            pixel_recall = getattr(metrics, "pixel_recall", 0.0)
        else:
            # Fallback - keine Metriken verfügbar
            iou_score = point_game_score = pixel_precision = pixel_recall = 0.0

        try:
            img_pil = Image.open(result.image_path)
            original_image = img_pil.resize((224, 224))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            attribution = result.attribution.numpy().squeeze(0)
            sns.heatmap(
                attribution,
                ax=axes[1],
                cmap="rocket",
                cbar=True,
                xticklabels=False,
                yticklabels=False,
                square=True,
            )
            axes[1].set_title(f"{result.explainer_name} Attribution", fontsize=12)

            axes[2].imshow(original_image)
            axes[2].imshow(attribution, cmap="rocket", alpha=0.4)

            if result.has_bbox:
                bbox_mask = bbox_to_mask_tensor(result.bbox)
                if bbox_mask is not None:
                    axes[2].contour(
                        bbox_mask.squeeze().numpy(),
                        levels=[0.5],
                        colors=sns.color_palette("husl", 1),
                        linewidths=2,
                    )

            axes[2].set_title("Attribution + BBox Overlay", fontsize=12)
            axes[2].axis("off")

            # KORRIGIERT: Verwende die extrahierten Metrik-Werte
            info_text = (
                f"Image: {result.image_name}\n"
                f"Model: {result.model_name}\n"
                f"Predicted: {result.predicted_class} "
                f"True: {result.true_label}\n"
                f"Correct: {'Yes' if result.prediction_correct else 'No'}\n"
            )

            # Füge Metriken hinzu falls verfügbar
            if (
                iou_score > 0
                or point_game_score > 0
                or pixel_precision > 0
                or pixel_recall > 0
            ):
                info_text += (
                    f"\nXAI Metrics:\n"
                    f"IoU: {iou_score:.3f}\n"
                    f"Point Game: {point_game_score:.3f}\n"
                    f"Pixel Precision: {pixel_precision:.3f}\n"
                    f"Pixel Recall: {pixel_recall:.3f}"
                )

            fig.suptitle(info_text, fontsize=10, ha="left")
            plt.tight_layout()

            save_path_str = None

            if self.save_path:
                # Falls self.save_path ein Ordner ist → baue Dateinamen dynamisch
                if self.save_path.is_dir():
                    vis_filename = (
                        f"{result.image_name}_{result.model_name}_"
                        f"{result.explainer_name}_vis.png"
                    )
                    vis_path = self.save_path / vis_filename
                else:
                    vis_path = self.save_path

                plt.savefig(vis_path, bbox_inches="tight", dpi=150)
                self.logger.info(f"Visualisierung gespeichert: {vis_path}")
                save_path_str = str(vis_path)

            if self.show:
                plt.show()
            else:
                plt.close()

            return save_path_str

        except ImportError:
            self.logger.warning(
                "Matplotlib oder Seaborn nicht verfügbar – keine Visualisierung"
            )
            return None
