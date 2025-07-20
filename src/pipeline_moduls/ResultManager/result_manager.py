import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import torch

from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from pipeline_moduls.evaluation.dataclass.evaluation_summary import EvaluationSummary
from pipeline_moduls.evaluation.dataclass.metricresults import MetricResults


class ResultManager:
    def __init__(self, attribution_dir="results/attributions"):
        self.results_per_step: Dict[str, List[XAIExplanationResult]] = {}
        self.dataframes_per_step: Dict[str, pd.DataFrame] = {}
        self.attribution_dir = attribution_dir
        self._logger = logging.getLogger(__name__)
        os.makedirs(self.attribution_dir, exist_ok=True)

    def add_results(self, step_name: str, new_results: List[XAIExplanationResult]):
        if step_name not in self.results_per_step:
            self.results_per_step[step_name] = []
        self.results_per_step[step_name].extend(new_results)

    def build_dataframe_for_step(self, step_name: str) -> pd.DataFrame:
        if (
            step_name not in self.results_per_step
            or not self.results_per_step[step_name]
        ):
            raise ValueError(f"No results found for step '{step_name}'.")
        records = []
        for r in self.results_per_step[step_name]:
            attribution_dir = os.path.join(
                self.attribution_dir, r.model_name, r.explainer_name
            )
            os.makedirs(attribution_dir, exist_ok=True)

            attribution_filename = f"{r.image_name}_attribution.pt"
            attribution_path = os.path.join(attribution_dir, attribution_filename)

            # Speichere Attribution als .pt file
            if r.attribution is not None:
                torch.save(r.attribution, attribution_path)

            # Konvertiere zu Dict und behandle alle Tensoren
            data = self._convert_result_to_csv_dict(r)
            data["attribution_path"] = attribution_path

            # Optional: direkt hier loggen
            if r.attribution is not None:
                mlflow.log_artifact(
                    attribution_path,
                    artifact_path=f"attributions/{r.model_name}/{r.explainer_name}",
                )

            records.append(data)
        df = pd.DataFrame(records)
        self.dataframes_per_step[step_name] = df
        return df

    def _convert_result_to_csv_dict(
        self, result: XAIExplanationResult
    ) -> Dict[str, Any]:
        """
        Konvertiert XAIExplanationResult zu einem CSV-freundlichen Dictionary.
        Behandelt Tensoren intelligent: entfernt sie oder konvertiert zu
        String-Repräsentation.
        """
        data = result.to_dict()

        # Liste der Felder die Tensoren enthalten können
        tensor_fields = ["image", "attribution", "bbox", "bbox_info"]

        # Behandle bekannte Tensor-Felder
        for field in tensor_fields:
            if field in data and data[field] is not None:
                if isinstance(data[field], torch.Tensor):
                    # Option 1: Entferne Tensor komplett
                    if field in ["image", "attribution"]:
                        del data[field]
                    # Option 2: Konvertiere zu String-Repräsentation für kleinere
                    # Tensoren
                    elif field in ["bbox", "bbox_info"]:
                        data[field] = self._tensor_to_string_representation(data[field])

        # Behandle explainer_result speziell
        if "explainer_result" in data and data["explainer_result"] is not None:
            data["explainer_result"] = self._convert_explainer_result_to_string(
                data["explainer_result"]
            )

        # Generische Tensor-Behandlung für unbekannte Felder
        data = self._clean_remaining_tensors(data)

        return data

    def _tensor_to_string_representation(self, tensor: torch.Tensor) -> str:
        """Konvertiert Tensor zu String-Repräsentation mit Shape und Typ Info"""
        if tensor is None:
            raise ValueError("Tensor is None")

        try:
            # Für kleine Tensoren: vollständige Werte
            if tensor.numel() <= 10:
                return f"<Tensor shape={tuple(tensor.shape)} values={tensor.tolist()}>"
            # Für größere Tensoren: nur Shape und Statistiken
            else:
                return (
                    f"<Tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                    f"device={tensor.device}>"
                )
        except Exception as e:
            self._logger.warning(f"Error converting tensor to string: {e}")
            return f"<Tensor shape={tuple(tensor.shape)}>"

    def _convert_explainer_result_to_string(self, explainer_result) -> str:
        """Konvertiert ExplainerResult zu String, ohne die großen Tensoren zu
        speichern"""
        if explainer_result is None:
            raise ValueError(f"explainer_result is none: {explainer_result}")

        try:
            # Erstelle summarische Darstellung
            summary_parts = []

            if (
                hasattr(explainer_result, "attributions")
                and explainer_result.attributions is not None
            ):
                attr_shape = tuple(explainer_result.attributions.shape)
                summary_parts.append(f"attributions_shape={attr_shape}")

            if (
                hasattr(explainer_result, "predictions")
                and explainer_result.predictions is not None
            ):
                pred_shape = tuple(explainer_result.predictions.shape)
                summary_parts.append(f"predictions_shape={pred_shape}")

            if (
                hasattr(explainer_result, "target_labels")
                and explainer_result.target_labels is not None
            ):
                target_shape = tuple(explainer_result.target_labels.shape)
                summary_parts.append(f"target_labels_shape={target_shape}")

            class_name = explainer_result.__class__.__name__
            return f"<{class_name}({', '.join(summary_parts)})>"

        except Exception as e:
            self._logger.warning(f"Error converting explainer_result to string: {e}")
            return f"<{type(explainer_result).__name__}>"

    def _clean_remaining_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Bereinigt verbleibende Tensoren aus dem Dictionary"""
        cleaned_data = {}

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Konvertiere unbekannte Tensoren zu String
                cleaned_data[key] = self._tensor_to_string_representation(value)
            elif isinstance(value, (list, tuple)):
                # Prüfe Listen/Tupel auf Tensoren
                cleaned_data[key] = self._clean_tensor_list(value)
            elif isinstance(value, dict):
                # Rekursiv für verschachtelte Dicts
                cleaned_data[key] = self._clean_remaining_tensors(value)
            else:
                # Behalte normale Werte
                cleaned_data[key] = value

        return cleaned_data

    def _clean_tensor_list(self, items):
        """Bereinigt Listen/Tupel die Tensoren enthalten können"""
        cleaned_items = []

        for item in items:
            if isinstance(item, torch.Tensor):
                cleaned_items.append(self._tensor_to_string_representation(item))
            elif isinstance(item, dict):
                cleaned_items.append(self._clean_remaining_tensors(item))
            elif isinstance(item, (list, tuple)):
                cleaned_items.append(self._clean_tensor_list(item))
            else:
                cleaned_items.append(item)

        return cleaned_items if isinstance(items, list) else tuple(cleaned_items)

    def add_evaluation_metrics_to_dataframe(
        self,
        step_name: str,
        evaluation_summary: EvaluationSummary,
        individual_metrics: Optional[List[MetricResults]] = None,
        evaluator=None,
    ) -> pd.DataFrame:
        """
        Erweitert das DataFrame um Evaluation-Metriken.

        Args:
            step_name: Name des Schrittes
            evaluation_summary: Summary der Evaluation
            individual_metrics: Optional - bereits berechnete individuelle Metriken
            evaluator: Optional - Evaluator um individuelle Metriken zu berechnen

        Returns:
            Erweitertes DataFrame mit Metrik-Spalten
        """
        self._logger.info(
            f"Adding evaluation metrics to DataFrame for step '{step_name}'"
        )

        # Hole oder erstelle DataFrame
        df = self.get_dataframe(step_name)

        # Methode 1: Verwende bereits berechnete individuelle Metriken
        if individual_metrics is not None:
            return self._add_individual_metrics_to_df(df, individual_metrics, step_name)

        # Methode 2: Berechne individuelle Metriken neu
        elif evaluator is not None:
            return self._calculate_and_add_metrics_to_df(df, evaluator, step_name)

        # Methode 3: Nur Summary-Metriken hinzufügen (weniger informativ)
        else:
            return self._add_summary_metrics_to_df(df, evaluation_summary)

    def _add_individual_metrics_to_df(
        self, df: pd.DataFrame, individual_metrics: List[MetricResults], step_name: str
    ) -> pd.DataFrame:
        """Füge individuelle Metriken zum DataFrame hinzu"""
        self._logger.info(
            f"Adding {len(individual_metrics)} individual metrics to DataFrame"
        )

        # Stelle sicher, dass wir die richtige Anzahl haben
        if len(individual_metrics) != len(df):
            self._logger.warning(
                f"Mismatch: {len(individual_metrics)} metrics vs {len(df)} DataFrame "
                "rows. Trying to match by image order."
            )

        # Initialisiere neue Spalten
        df["iou_score"] = None
        df["pixel_precision"] = None
        df["pixel_recall"] = None
        df["point_game_score"] = None
        df["evaluation_metrics_json"] = None

        # Fülle Metriken ein
        for i, metrics in enumerate(individual_metrics):
            if i >= len(df):
                break

            if metrics and metrics.values:
                row_idx = i

                # IoU Score
                if "IoU" in metrics.values:
                    df.loc[row_idx, "iou_score"] = float(metrics.values["IoU"])

                # Pixel Precision/Recall
                if "PixelPrecisionRecall" in metrics.values:
                    ppr = metrics.values["PixelPrecisionRecall"]
                    if isinstance(ppr, dict):
                        df.loc[row_idx, "pixel_precision"] = float(
                            ppr.get("precision", 0)
                        )
                        df.loc[row_idx, "pixel_recall"] = float(ppr.get("recall", 0))
                    else:
                        df.loc[row_idx, "pixel_precision"] = float(ppr)

                # Point Game
                if "point_game" in metrics.values:
                    df.loc[row_idx, "point_game_score"] = float(
                        metrics.values["point_game"]
                    )

                # Alle Metriken als JSON
                import json

                df.loc[row_idx, "evaluation_metrics_json"] = json.dumps(
                    {
                        k: (v if not isinstance(v, torch.Tensor) else v.item())
                        for k, v in metrics.values.items()
                    }
                )

        # Aktualisiere gespeicherten DataFrame
        self.dataframes_per_step[step_name] = df
        self._logger.info("Individual metrics successfully added to DataFrame")
        return df

    def _calculate_and_add_metrics_to_df(
        self, df: pd.DataFrame, evaluator, step_name: str
    ) -> pd.DataFrame:
        """Berechne Metriken neu und füge sie zum DataFrame hinzu"""
        self._logger.info("Calculating individual metrics for DataFrame rows")

        # Lade Results für diesen Step
        results = self.results_per_step.get(step_name, [])
        if not results:
            self._logger.warning(
                f"No results found for step '{step_name}' - cannot calculate metrics"
            )
            return df

        # Berechne individuelle Metriken
        individual_metrics = []
        for result in results:
            metrics = evaluator.evaluate_single_result(result)
            individual_metrics.append(metrics)

        # Verwende die individuelle Metriken Methode
        return self._add_individual_metrics_to_df(df, individual_metrics, step_name)

    def _add_summary_metrics_to_df(
        self, df: pd.DataFrame, evaluation_summary: EvaluationSummary
    ) -> pd.DataFrame:
        """Füge nur Summary-Metriken hinzu (alle Zeilen bekommen die gleichen Werte)"""
        self._logger.info("Adding summary metrics to all DataFrame rows")

        # Füge Summary-Metriken als neue Spalten hinzu
        df["prediction_accuracy_summary"] = evaluation_summary.prediction_accuracy
        df["average_processing_time_summary"] = (
            evaluation_summary.average_processing_time
        )
        df["samples_with_bbox_summary"] = evaluation_summary.samples_with_bbox

        # Füge durchschnittliche Metrik-Werte hinzu
        if evaluation_summary.metric_averages:
            for metric_name, avg_value in evaluation_summary.metric_averages.items():
                df[f"{metric_name}_summary"] = avg_value

        return df

    def save_dataframe_with_metrics(
        self,
        step_name: str,
        path: str,
        evaluation_summary: Optional[EvaluationSummary] = None,
        individual_metrics: Optional[List[MetricResults]] = None,
        evaluator=None,
    ):
        """
        Speichert das DataFrame mit Evaluation-Metriken als CSV.

        Args:
            step_name: Name des Schrittes
            path: Pfad zum Speichern
            evaluation_summary: Optional - Evaluation Summary
            individual_metrics: Optional - Individuelle Metriken
            evaluator: Optional - Evaluator für neue Berechnung
        """
        # Erweitere DataFrame um Metriken falls verfügbar
        if evaluation_summary or individual_metrics or evaluator:
            df = self.add_evaluation_metrics_to_dataframe(
                step_name=step_name,
                evaluation_summary=evaluation_summary,
                individual_metrics=individual_metrics,
                evaluator=evaluator,
            )
        else:
            df = self.get_dataframe(step_name)

        # Speichere CSV
        df.to_csv(path, index=False)
        self._logger.info(f"DataFrame with metrics saved to {path}")

    def get_dataframe(self, step_name: str) -> pd.DataFrame:
        """
        Gibt den DataFrame eines Schrittes zurück, baut ihn bei Bedarf.
        """
        if step_name not in self.dataframes_per_step:
            return self.build_dataframe_for_step(step_name)
        return self.dataframes_per_step[step_name]

    def save_dataframe(self, step_name: str, path: str):
        """
        Speichert den DataFrame eines Schrittes als CSV.
        """
        df = self.get_dataframe(step_name)
        df.to_csv(path, index=False)

    def reset(self):
        """
        Setzt alle gespeicherten Ergebnisse und DataFrames zurück.
        """
        self.results_per_step.clear()
        self.dataframes_per_step.clear()

    def save_results(self, results: List[XAIExplanationResult], path: Path):
        torch.save(results, path)
