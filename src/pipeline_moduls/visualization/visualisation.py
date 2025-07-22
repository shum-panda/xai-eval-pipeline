import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults
from src.pipeline_moduls.evaluation.xai_evaluator import bbox_to_mask_tensor


class Visualiser:
    def __init__(self, show: bool = True, save_path: Optional[Path] = None):
        """
        Visualizer for displaying XAI results.

        Args:
            show: Whether to display the visualization.
            save_path: Optional path for saving the plot.
        """
        self.show = show
        self.save_path = save_path
        self.logger = logging.getLogger(__name__)

        # Activate seaborn style
        sns.set_theme(style="whitegrid")

    def create_visualization(
        self,
        result: XAIExplanationResult,
        metrics: Union[EvaluationSummary, MetricResults, None] = None,
    ) -> Optional[str]:
        """
        Creates a visualization for an XAI result with 2x2 grid layout.

        Args:
            result: XAI Explanation Result
            metrics: Can be EvaluationSummary, individual Metrics object, or None

        Returns:
            Optional path to saved visualization file
        """

        # Extract individual metric values (not averages!)
        iou_score, point_game_score, pixel_precision, pixel_recall = (
            self._extract_individual_metrics(metrics)
        )

        try:
            # Load and scale image
            img_pil = Image.open(result.image_path)
            original_image = img_pil.resize((224, 224))

            # Create 2x2 grid layout with minimal whitespace
            fig = plt.figure(figsize=(12, 10))

            # Define grid with minimal whitespace between columns
            gs = fig.add_gridspec(
                2, 2, hspace=0.0, wspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0
            )

            # Quadrant 1 (top left): Metric values as text
            ax_metrics = fig.add_subplot(gs[0, 0])
            self._create_metrics_display(
                ax_metrics,
                result,
                iou_score,
                point_game_score,
                pixel_precision,
                pixel_recall,
            )

            # Quadrant 2 (top right): Original image
            ax_original = fig.add_subplot(gs[0, 1])
            ax_original.imshow(original_image)
            ax_original.set_title("Original Image", fontsize=14, pad=5)
            ax_original.axis("off")

            # Quadrant 3 (bottom left): Attribution heatmap
            ax_attribution = fig.add_subplot(gs[1, 0])
            attribution = result.attribution.numpy().squeeze(0)
            sns.heatmap(
                attribution,
                ax=ax_attribution,
                cmap="rocket",
                cbar=True,
                xticklabels=False,
                yticklabels=False,
                square=True,
                cbar_kws={"shrink": 0.7, "pad": 0.02},  # Smaller colorbar, less padding
            )
            ax_attribution.text(
                0.5,
                -0.01,
                f"{result.explainer_name} Attribution",
                transform=ax_attribution.transAxes,
                fontsize=14,
                ha="center",
                va="top",
            )

            # Quadrant 4 (bottom right): Overlay with BBox
            ax_overlay = fig.add_subplot(gs[1, 1])
            ax_overlay.imshow(original_image)
            ax_overlay.imshow(attribution, cmap="rocket", alpha=0.4)

            # Add BBox contour if available
            if result.has_bbox:
                bbox_mask = bbox_to_mask_tensor(result.bbox)
                if bbox_mask is not None:
                    ax_overlay.contour(
                        bbox_mask.squeeze().numpy(),
                        levels=[0.5],
                        colors=["lime"],  # Light green for better visibility
                        linewidths=2,
                    )

            ax_overlay.text(
                0.5,
                -0.01,
                "Attribution + BBox Overlay",
                transform=ax_overlay.transAxes,
                fontsize=14,
                ha="center",
                va="top",
            )

            ax_overlay.axis("off")

            # Main title with compact info (closer to plot)
            main_title = (
                f"{result.image_name} | {result.model_name} | {result.explainer_name}"
            )
            fig.suptitle(main_title, fontsize=14, y=1.05)

            # Remove additional whitespace around plot
            plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.00)

            # Save and show plot
            save_path_str = self._save_and_show_plot(fig, result)
            return save_path_str

        except ImportError:
            self.logger.warning(
                "Matplotlib or Seaborn not available – no visualization"
            )
            return None
        except Exception as e:
            self.logger.error(f"Error during visualization: {e}")
            return None

    def _extract_individual_metrics(self, metrics):
        """Extracts individual metric values (not average values)"""

        if isinstance(metrics, MetricResults):
            # Individual metrics - this is what we want!
            return self._extract_from_metric_results(metrics)
        elif hasattr(metrics, "metric_averages"):
            # EvaluationSummary - use average values as fallback
            self.logger.warning(
                "Using average values since no individual metrics available"
            )
            iou_score = metrics.metric_averages.get("average_IoU", 0.0)
            point_game_score = metrics.metric_averages.get("average_point_game", 0.0)
            pixel_precision = metrics.metric_averages.get(
                "average_PixelPrecisionRecall_precision", 0.0
            )
            pixel_recall = metrics.metric_averages.get(
                "average_PixelPrecisionRecall_recall", 0.0
            )
            return iou_score, point_game_score, pixel_precision, pixel_recall
        else:
            # No metrics available
            self.logger.info("No metrics available for visualization")
            return 0.0, 0.0, 0.0, 0.0

    def _extract_from_metric_results(self, metrics: MetricResults):
        """Extracts values from MetricResults object"""

        # MetricResults has a .values dictionary with the actual values
        if hasattr(metrics, "values") and isinstance(metrics.values, dict):
            values = metrics.values

            # IoU Score
            iou_score = values.get("IoU", 0.0)

            # Point Game Score
            point_game_score = values.get("point_game", 0.0)

            # Pixel Precision/Recall (can be nested)
            pixel_metrics = values.get("PixelPrecisionRecall", {})
            if isinstance(pixel_metrics, dict):
                pixel_precision = pixel_metrics.get("precision", 0.0)
                pixel_recall = pixel_metrics.get("recall", 0.0)
            else:
                pixel_precision = 0.0
                pixel_recall = 0.0

            return iou_score, point_game_score, pixel_precision, pixel_recall
        else:
            # Fallback: try direct attributes
            iou_score = getattr(metrics, "iou_score", 0.0)
            point_game_score = getattr(metrics, "point_game_score", 0.0)
            pixel_precision = getattr(metrics, "pixel_precision", 0.0)
            pixel_recall = getattr(metrics, "pixel_recall", 0.0)

            return iou_score, point_game_score, pixel_precision, pixel_recall

    def _create_metrics_display(
        self, ax, result, iou_score, point_game_score, pixel_precision, pixel_recall
    ):
        """Creates a clean metrics display in the top left quadrant"""

        ax.axis("off")  # No axes

        # Basic information
        info_lines = [
            f"Image: {result.image_name}",
            f"Model: {result.model_name}",
            f"Explainer: {result.explainer_name}",
            "",  # Empty line
            f"Predicted: {result.predicted_class}",
            f"True Label: {result.true_label}",
            f"Correct: {'[+] Yes' if result.prediction_correct else '[-] No'}",
        ]

        # Add XAI metrics if available
        if any(
            [iou_score > 0, point_game_score > 0, pixel_precision > 0, pixel_recall > 0]
        ):
            info_lines.extend(
                [
                    "",  # Empty line
                    "XAI Metrics:",
                    f"IoU: {iou_score:.3f}",
                    f"Point Game: {point_game_score:.3f}",
                    f"Pixel Precision: {pixel_precision:.3f}",
                    f"Pixel Recall: {pixel_recall:.3f}",
                ]
            )

        # Display text centered in quadrant
        text_content = "\n".join(info_lines)
        ax.text(
            0.05,
            0.90,
            text_content,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="left",
        )

    def _save_and_show_plot(self, fig, result):
        """Saves and shows the plot"""
        save_path_str = None

        if self.save_path:
            # If self.save_path is a directory → build filename dynamically
            if self.save_path.is_dir():
                vis_filename = (
                    f"{result.image_name}_{result.model_name}_"
                    f"{result.explainer_name}_vis.png"
                )
                vis_path = self.save_path / vis_filename
            else:
                vis_path = self.save_path

            plt.savefig(vis_path, bbox_inches="tight", dpi=150, facecolor="white")
            self.logger.info(f"Visualization saved: {vis_path}")
            save_path_str = str(vis_path)

        if self.show:
            plt.show()
        else:
            plt.close()

        return save_path_str
