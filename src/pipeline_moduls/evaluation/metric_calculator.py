import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from pipeline_moduls.evaluation.dataclass.evaluation_summary import EvaluationSummary
from pipeline_moduls.evaluation.dataclass.xai_metrics import XAIMetrics

class XAIEvaluator:
    """
    Evaluiert XAI Ergebnisse mit verschiedenen Metriken
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info("XAI Evaluator initialisiert")

    def evaluate_single_result(
        self,
        result: XAIExplanationResult,
        pointing_threshold: float = 0.15,
        iou_threshold: float = 0.5,
        coverage_percentile: float = 90,
    ) -> Optional[XAIMetrics]:
        """
        Evaluiere ein einzelnes XAI Ergebnis

        Args:
            result: XAI Explanation Result vom Orchestrator
            pointing_threshold: Threshold für Pointing Game (Top %)
            iou_threshold: Threshold für IoU Berechnung
            coverage_percentile: Percentile für Coverage Berechnung

        Returns:
            XAI Metriken oder None falls keine BBox
        """
        if not result.has_bbox or result.bbox_info is None:
            return None

        # Erstelle Bounding Box Mask
        bbox_mask = bbox_to_mask_tensor(result.bbox)
        if bbox_mask is None or torch.sum(bbox_mask) == 0:
            return None

        attribution = result.attribution

        # Pointing Game
        pointing_metrics = self._compute_pointing_game(
            attribution, bbox_mask, pointing_threshold
        )

        # IoU Score
        iou_score = self._compute_iou(attribution, bbox_mask, iou_threshold)

        # Coverage Score
        coverage_score = self._compute_coverage(
            attribution, bbox_mask, coverage_percentile
        )

        # Precision/Recall
        precision, recall = self._compute_precision_recall(
            attribution, bbox_mask, iou_threshold
        )

        return XAIMetrics(
            pointing_game_hit=pointing_metrics["hit"] == 1,
            pointing_game_threshold=pointing_threshold,
            iou_score=iou_score,
            iou_threshold=iou_threshold,
            coverage_score=coverage_score,
            coverage_percentile=coverage_percentile,
            intersection_area=pointing_metrics["intersection"],
            bbox_area=pointing_metrics["bbox_area"],
            attribution_area=pointing_metrics["attr_area"],
            precision=precision,
            recall=recall,
        )

    def evaluate_batch_results(
        self, results: Union[List[XAIExplanationResult], pd.DataFrame], **metric_kwargs
    ) -> EvaluationSummary:
        """
        Evaluiere eine Liste oder DataFrame von XAI Ergebnissen

        Args:
            results: Liste von XAIExplanationResult ODER DataFrame
            **metric_kwargs: Parameter für Metriken

        Returns:
            EvaluationSummary
        """
        if isinstance(results, pd.DataFrame):
            if results.empty:
                raise ValueError("Leerer DataFrame übergeben")
            # Konvertiere DataFrame → List[XAIExplanationResult]
            results = [
                XAIExplanationResult.from_dict(row.to_dict())
                for _, row in results.iterrows()
            ]

        if not results:
            raise ValueError("Keine Ergebnisse zum Evaluieren")

        self._logger.info(f"Evaluiere {len(results)} Ergebnisse...")

        metrics_list = []
        correct_predictions = 0
        total_processing_time = 0

        for result in results:
            # Prediction Accuracy
            if result.prediction_correct is not None and result.prediction_correct:
                correct_predictions += 1

            total_processing_time += result.processing_time

            # XAI Metriken
            metrics = self.evaluate_single_result(result, **metric_kwargs)
            if metrics:
                metrics_list.append(metrics)

        summary = self._aggregate_metrics(
            results=results,
            metrics_list=metrics_list,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

        self._log_summary(summary)
        return summary

    def compare_explainers(
        self, explainer_results: Dict[str, List[XAIExplanationResult]], **metric_kwargs
    ) -> Dict[str, EvaluationSummary]:
        """
        Vergleiche mehrere Explainer

        Args:
            explainer_results: Dict mit Explainer Namen -> Results
            **metric_kwargs: Parameter für Metriken

        Returns:
            Dict mit Evaluation Summaries pro Explainer
        """
        comparison_results = {}

        for explainer_name, results in explainer_results.items():
            self._logger.info(f"Evaluiere {explainer_name}...")

            try:
                summary = self.evaluate_batch_results(results, **metric_kwargs)
                comparison_results[explainer_name] = summary
            except Exception as e:
                self._logger.error(f"Fehler bei {explainer_name}: {e}")
                continue

        # Erstelle Vergleichstabelle
        self._create_comparison_table(comparison_results)

        return comparison_results

    def save_evaluation_results(
        self,
        summary: EvaluationSummary,
        detailed_results: Optional[List[XAIExplanationResult]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Speichere Evaluation Ergebnisse

        Args:
            summary: Evaluation Summary
            detailed_results: Detaillierte Ergebnisse (optional)
            output_dir: Output Directory
        """
        if output_dir is None:
            output_dir = Path("")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary speichern
        summary_file = (
            output_dir / f"evaluation_summary_{summary.explainer_name}_{timestamp}.json"
        )

        summary_dict = {
            "explainer_name": summary.explainer_name,
            "model_name": summary.model_name,
            "total_samples": summary.total_samples,
            "samples_with_bbox": summary.samples_with_bbox,
            "prediction_accuracy": summary.prediction_accuracy,
            "pointing_game_score": summary.pointing_game_score,
            "average_iou": summary.average_iou,
            "average_coverage": summary.average_coverage,
            "average_precision": summary.average_precision,
            "average_recall": summary.average_recall,
            "average_processing_time": summary.average_processing_time,
            "evaluation_timestamp": summary.evaluation_timestamp,
        }

        with open(summary_file, "w") as f:
            json.dump(summary_dict, f, indent=2)

        self._logger.info(f"Summary gespeichert: {summary_file}")

        # Detaillierte Ergebnisse (optional)
        if detailed_results:
            details_file = (
                output_dir
                / f"evaluation_details_{summary.explainer_name}_{timestamp}.json"
            )
            details_dict = self._serialize_detailed_results(detailed_results)

            with open(details_file, "w") as f:
                json.dump(details_dict, f, indent=2)

            self._logger.info(f"Details gespeichert: {details_file}")

    def save_comparison_results(
        self,
        comparison_results: Dict[str, EvaluationSummary],
        output_dir: Optional[Path] = None,
    ):
        """
        Speichere Explainer Vergleich

        Args:
            comparison_results: Vergleichsergebnisse
            output_dir: Output Directory
        """
        if output_dir is None:
            output_dir = Path("")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = output_dir / f"explainer_comparison_{timestamp}.json"

        # Konvertiere zu serializable format
        comparison_dict = {}
        for explainer_name, summary in comparison_results.items():
            comparison_dict[explainer_name] = {
                "explainer_name": summary.explainer_name,
                "model_name": summary.model_name,
                "total_samples": summary.total_samples,
                "samples_with_bbox": summary.samples_with_bbox,
                "prediction_accuracy": summary.prediction_accuracy,
                "pointing_game_score": summary.pointing_game_score,
                "average_iou": summary.average_iou,
                "average_coverage": summary.average_coverage,
                "average_precision": summary.average_precision,
                "average_recall": summary.average_recall,
                "average_processing_time": summary.average_processing_time,
            }

        with open(comparison_file, "w") as f:
            json.dump(comparison_dict, f, indent=2)

        self._logger.info(f"Vergleich gespeichert: {comparison_file}")

    def _compute_pointing_game(
        self, attribution: torch.Tensor, bbox_mask: torch.Tensor, threshold: float
    ) -> Dict[str, float]:
        """Berechne Pointing Game Metrik"""
        # Top threshold% der Attribution
        flat_attr = attribution.flatten()
        num_top = max(1, int(len(flat_attr) * threshold))
        threshold_val = torch.topk(flat_attr, num_top)[0][-1]

        top_attr_mask = (attribution >= threshold_val).float()

        # Intersection mit BBox
        intersection = torch.sum(top_attr_mask * bbox_mask)
        bbox_area = torch.sum(bbox_mask)
        attr_area = torch.sum(top_attr_mask)

        # Hit = mindestens 1 Pixel Overlap
        hit = intersection > 0

        return {
            "hit": hit.item(),
            "intersection": intersection.item(),
            "bbox_area": bbox_area.item(),
            "attr_area": attr_area.item(),
        }

    def _compute_iou(
        self, attribution: torch.Tensor, bbox_mask: torch.Tensor, threshold: float
    ) -> float:
        """Berechne IoU Score"""
        # Binarisiere Attribution
        attr_binary = (attribution >= threshold).float()

        # IoU
        intersection = torch.sum(attr_binary * bbox_mask)
        union = torch.sum(attr_binary) + torch.sum(bbox_mask) - intersection

        iou = (intersection / union) if union > 0 else 0
        return iou.item()

    def _compute_coverage(
        self, attribution: torch.Tensor, bbox_mask: torch.Tensor, percentile: float
    ) -> float:
        """Berechne Coverage Score"""
        # Top percentile der Attribution

        threshold = attribution.quantile(percentile / 100.0)
        top_attr_mask = (attribution >= threshold).float()

        # Coverage der BBox
        intersection = torch.sum(top_attr_mask * bbox_mask)
        bbox_area = torch.sum(bbox_mask)

        coverage = (intersection / bbox_area) if bbox_area > 0 else 0
        return coverage.item()

    def _compute_precision_recall(
        self, attribution: torch.Tensor, bbox_mask: torch.Tensor, threshold: float
    ) -> tuple:
        """Berechne Precision und Recall"""
        attr_binary = (attribution >= threshold).float()

        # True Positives, False Positives, False Negatives
        tp = torch.sum(attr_binary * bbox_mask)
        fp = torch.sum(attr_binary * (1.0 - bbox_mask))
        fn = torch.sum((1 - attr_binary) * bbox_mask)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0

        return precision, recall

    def _aggregate_metrics(
        self,
        results: List[XAIExplanationResult],
        metrics_list: List[XAIMetrics],
        correct_predictions: int,
        total_processing_time: float,
    ) -> EvaluationSummary:
        """Aggregiere Metriken zu Summary"""

        # Basic Info
        explainer_name = results[0].explainer_name if results else "unknown"
        model_name = results[0].model_name if results else "unknown"

        # Averages
        if metrics_list:
            pointing_hits = sum(1 for m in metrics_list if m.pointing_game_hit)
            pointing_game_score = pointing_hits / len(metrics_list)

            average_iou = np.mean([m.iou_score for m in metrics_list])
            average_coverage = np.mean([m.coverage_score for m in metrics_list])
            average_precision = np.mean([m.precision for m in metrics_list])
            average_recall = np.mean([m.recall for m in metrics_list])
        else:
            pointing_game_score = average_iou = average_coverage = 0
            average_precision = average_recall = 0

        # Prediction accuracy
        prediction_accuracy = correct_predictions / len(results) if results else 0

        # Processing time
        average_processing_time = total_processing_time / len(results) if results else 0

        return EvaluationSummary(
            explainer_name=explainer_name,
            model_name=model_name,
            total_samples=len(results),
            samples_with_bbox=len(metrics_list),
            prediction_accuracy=float(prediction_accuracy),
            correct_predictions=int(correct_predictions),
            pointing_game_score=float(pointing_game_score),
            average_iou=float(average_iou),
            average_coverage=float(average_coverage),
            average_precision=float(average_precision),
            average_recall=float(average_recall),
            average_processing_time=float(average_processing_time),
            total_processing_time=float(total_processing_time),
            evaluation_timestamp=datetime.now().isoformat(),
        )

    def _log_summary(self, summary: EvaluationSummary):
        """Logge Evaluation Summary"""
        self._logger.info(f"Evaluation Summary für {summary.explainer_name}:")
        self._logger.info(f"  Total Samples: {summary.total_samples}")
        self._logger.info(f"  Samples with BBox: {summary.samples_with_bbox}")
        self._logger.info(f"  Prediction Accuracy: {summary.prediction_accuracy:.3f}")
        self._logger.info(f"  Pointing Game Score: {summary.pointing_game_score:.3f}")
        self._logger.info(f"  Average IoU: {summary.average_iou:.3f}")
        self._logger.info(f"  Average Coverage: {summary.average_coverage:.3f}")
        self._logger.info(
            f"  Average Processing Time: {summary.average_processing_time:.3f}s"
        )

    def _create_comparison_table(
        self, comparison_results: Dict[str, EvaluationSummary]
    ):
        """Erstelle Vergleichstabelle"""
        if not comparison_results:
            return

        self._logger.info(f"\n{'=' * 80}")
        self._logger.info(f"EXPLAINER COMPARISON")
        self._logger.info(f"{'=' * 80}")

        # Header
        header = f"{'Explainer':<15} {'Pred Acc':<10} {'Point Game':<12} {'Avg IoU':<10} {'Avg Cov':<10} {'Avg Time':<10}"
        self._logger.info(header)
        self._logger.info(f"{'-' * 80}")

        # Rows
        for explainer_name, summary in comparison_results.items():
            row = f"{explainer_name:<15} "
            row += f"{summary.prediction_accuracy:<10.3f} "
            row += f"{summary.pointing_game_score:<12.3f} "
            row += f"{summary.average_iou:<10.3f} "
            row += f"{summary.average_coverage:<10.3f} "
            row += f"{summary.average_processing_time:<10.3f}"
            self._logger.info(row)

        self._logger.info(f"{'=' * 80}")

    def _serialize_detailed_results(
        self, results: List[XAIExplanationResult]
    ) -> List[Dict]:
        """Serialisiere detaillierte Ergebnisse für JSON"""
        serialized = []

        for result in results:
            item = {
                # todo Imagepath
                "image_name": result.image_name,
                "predicted_class": result.predicted_class,
                "true_label": result.true_label,
                "prediction_correct": result.prediction_correct,
                "explainer_name": result.explainer_name,
                "model_name": result.model_name,
                "has_bbox": result.has_bbox,
                "processing_time": result.processing_time,
                # Note: attribution und explainer_result werden nicht serialisiert (zu groß)
            }
            serialized.append(item)

        return serialized


def bbox_to_mask_tensor(bbox, shape=(224, 224)):
    """
    Wandelt eine einzelne Bounding Box (Tensor [1, 4]) in eine Binärmaske [1, H, W] um.
    """
    mask = torch.zeros((1, *shape), dtype=torch.float32, device=bbox.device)
    x1, y1, x2, y2 = bbox[0].int()  # [1, 4] → [4]
    mask[0, y1:y2, x1:x2] = 1.0
    return mask
