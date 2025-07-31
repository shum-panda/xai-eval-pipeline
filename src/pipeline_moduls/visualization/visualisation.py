import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

from pipeline_moduls.utils.bbox_to_mask_tensor import bbox_to_mask_tensor
from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults


class Visualiser:
    def __init__(self, show: bool = True, save_path: Optional[Path] = None) -> None:
        """
        Visualiser for displaying XAI results.

        Args:
            show (bool): Whether to display the visualization after creation.
            save_path (Optional[Path]): Path to save the visualization image. Can be
            a directory or a file path.
        """
        self.show: bool = show
        self.save_path: Optional[Path] = save_path
        self.logger: logging.Logger = logging.getLogger(__name__)

        sns.set_theme(style="whitegrid")

    def create_visualization(
        self,
        result: XAIExplanationResult,
        metrics: Optional[Union[EvaluationSummary, MetricResults]] = None,
    ) -> Optional[str]:
        """
        Create a 2x2 grid visualization for the given XAI explanation result.

        The layout includes:
          - Top-left: Metrics text info
          - Top-right: Original image
          - Bottom-left: Attribution heatmap
          - Bottom-right: Overlay of attribution and bounding box (if available)

        Args:
            result (XAIExplanationResult): The explanation result containing image,
            attribution, and metadata.
            metrics (Optional[Union[EvaluationSummary, MetricResults]]): Metrics
            related to the explanation, for display.

        Returns:
            Optional[str]: Path to the saved visualization image file, or None if not
            saved.
        """
        iou_score, point_game_score, pixel_precision, pixel_recall = (
            self._extract_individual_metrics(metrics)
        )

        try:
            img_pil: Image.Image = Image.open(result.image_path)
            original_image = img_pil.resize((224, 224))

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(
                2, 2, hspace=0.0, wspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0
            )

            ax_metrics = fig.add_subplot(gs[0, 0])
            self._create_metrics_display(
                ax_metrics,
                result,
                iou_score,
                point_game_score,
                pixel_precision,
                pixel_recall,
            )

            ax_original = fig.add_subplot(gs[0, 1])
            ax_original.imshow(original_image)
            ax_original.set_title("Original Image", fontsize=14, pad=5)
            ax_original.axis("off")

            ax_attribution = fig.add_subplot(gs[1, 0])
            attribution_np: np.ndarray = self._prepare_attribution_for_heatmap(
                result.attribution
            )

            sns.heatmap(
                attribution_np,
                ax=ax_attribution,
                cmap="rocket",
                cbar=True,
                xticklabels=False,
                yticklabels=False,
                square=True,
                cbar_kws={"shrink": 0.7, "pad": 0.02},
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

            ax_overlay = fig.add_subplot(gs[1, 1])
            ax_overlay.imshow(original_image)
            ax_overlay.imshow(attribution_np, cmap="rocket", alpha=0.4)

            if result.has_bbox:
                bbox_mask = bbox_to_mask_tensor(result.bbox)
                if bbox_mask is not None:
                    contour_data = bbox_mask.squeeze().cpu().numpy()
                    ax_overlay.contour(
                        contour_data,
                        levels=[0.5],
                        colors=["lime"],
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

            main_title = (
                f"{result.image_name} | {result.model_name} | {result.explainer_name}"
            )
            fig.suptitle(main_title, fontsize=14, y=1.05)

            plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.00)

            save_path_str = self._save_and_show_plot(fig, result)
            return save_path_str

        except ImportError:
            self.logger.warning(
                "Matplotlib or Seaborn not available – no visualization"
            )
            return None
        except Exception as e:
            self.logger.error(f"Error during visualization: {e}")
            raise

    def _prepare_attribution_for_heatmap(self, attribution: torch.Tensor) -> np.ndarray:
        if attribution.dim() > 2:
            attribution = attribution.mean(dim=0)
        attribution_np = attribution.detach().cpu().numpy()

        # Optional: Absolutwert (besonders für IG sinnvoll)
        attribution_np = np.abs(attribution_np)

        # Normalisieren
        attribution_np -= attribution_np.min()
        max_val = attribution_np.max()
        if max_val != 0:
            attribution_np /= max_val

        return attribution_np

    def _extract_individual_metrics(
        self, metrics: Optional[Union[EvaluationSummary, MetricResults]]
    ) -> Tuple[float, float, float, float]:
        """
        Extract individual metric values from the provided metrics object.

        Args:
            metrics (Optional[Union[EvaluationSummary, MetricResults]]): Metrics
            object or None.

        Returns:
            Tuple[float, float, float, float]: IoU score, Point Game score,
            Pixel Precision, Pixel Recall.
        """
        if isinstance(metrics, MetricResults):
            return self._extract_from_metric_results(metrics)
        elif metrics is not None and hasattr(metrics, "metric_averages"):
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
            self.logger.info("No metrics available for visualization")
            return 0.0, 0.0, 0.0, 0.0

    def _extract_from_metric_results(
        self, metrics: MetricResults
    ) -> Tuple[float, float, float, float]:
        """
        Extract metric values from a MetricResults object.

        Args:
            metrics (MetricResults): MetricResults object.

        Returns:
            Tuple[float, float, float, float]: IoU score, Point Game score,
            Pixel Precision, Pixel Recall.
        """
        if hasattr(metrics, "values") and isinstance(metrics.values, dict):
            values = metrics.values
            iou_score = values.get("IoU", 0.0)
            point_game_score = values.get("point_game", 0.0)

            pixel_metrics = values.get("PixelPrecisionRecall", {})
            if isinstance(pixel_metrics, dict):
                pixel_precision = pixel_metrics.get("precision", 0.0)
                pixel_recall = pixel_metrics.get("recall", 0.0)
            else:
                pixel_precision = 0.0
                pixel_recall = 0.0

            return iou_score, point_game_score, pixel_precision, pixel_recall
        else:
            iou_score = getattr(metrics, "iou_score", 0.0)
            point_game_score = getattr(metrics, "point_game_score", 0.0)
            pixel_precision = getattr(metrics, "pixel_precision", 0.0)
            pixel_recall = getattr(metrics, "pixel_recall", 0.0)

            return iou_score, point_game_score, pixel_precision, pixel_recall

    def _create_metrics_display(
        self,
        ax: plt.Axes,
        result: XAIExplanationResult,
        iou_score: float,
        point_game_score: float,
        pixel_precision: float,
        pixel_recall: float,
    ) -> None:
        """
        Create a clean, text-based display of metrics inside a matplotlib Axes.

        Args:
            ax (plt.Axes): Matplotlib axis where the text will be drawn.
            result (XAIExplanationResult): Explanation result with metadata.
            iou_score (float): Intersection over Union score.
            point_game_score (float): Point Game score.
            pixel_precision (float): Pixel Precision score.
            pixel_recall (float): Pixel Recall score.
        """
        ax.axis("off")

        # Nutze die String-Labels aus dem Result, falls vorhanden, sonst "N/A"
        predicted_label_str = result.predicted_class_name or "N/A"
        true_label_str = result.true_label_name or "N/A"

        info_lines = [
            f"Image: {result.image_name}",
            f"Model: {result.model_name}",
            f"Explainer: {result.explainer_name}",
            "",
            f"Predicted: {result.predicted_class} ({predicted_label_str})",
            f"True Label: {result.true_label} ({true_label_str})",
            f"Correct: {'[+] Yes' if result.prediction_correct else '[-] No'}",
        ]

        info_lines.extend(
            [
                "",
                "XAI Metrics:",
                f"IoU: {iou_score:.3f}",
                f"Point Game: {point_game_score:.3f}",
                f"Pixel Precision: {pixel_precision:.3f}",
                f"Pixel Recall: {pixel_recall:.3f}",
            ]
        )

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

    def _save_and_show_plot(
        self, fig: plt.Figure, result: XAIExplanationResult
    ) -> Optional[str]:
        """
        Save the generated figure to disk if a save path is provided and show or
        close the figure.

        Args:
            fig (plt.Figure): Matplotlib figure object.
            result (XAIExplanationResult): Explanation result for naming the file.

        Returns:
            Optional[str]: The path to the saved file or None if not saved.
        """
        save_path_str: Optional[str] = None

        if self.save_path:
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
        plt.close(fig)

        return save_path_str
