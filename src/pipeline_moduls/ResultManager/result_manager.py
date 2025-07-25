import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import torch

from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.dataclass.metricresults import MetricResults


class ResultManager:
    """
    Manages the collection, storage, and processing of XAI explanation results,
    including saving attributions, building dataframes, and adding evaluation metrics.
    """

    def __init__(self, attribution_dir: str = "results/attributions") -> None:
        """
        Initializes the ResultManager.

        Args:
            attribution_dir (str): Directory where attribution files will be saved.
        """
        self.results_per_step: Dict[str, List[XAIExplanationResult]] = {}
        self.dataframes_per_step: Dict[str, pd.DataFrame] = {}
        self.attribution_dir = attribution_dir
        self._logger = logging.getLogger(__name__)
        os.makedirs(self.attribution_dir, exist_ok=True)

    def add_results(
        self, step_name: str, new_results: List[XAIExplanationResult]
    ) -> None:
        """
        Adds new XAI explanation results for a given processing step.

        Args:
            step_name (str): Name of the processing step.
            new_results (List[XAIExplanationResult]): List of explanation results to
            add.
        """
        if step_name not in self.results_per_step:
            self.results_per_step[step_name] = []
        self.results_per_step[step_name].extend(new_results)

    def build_dataframe_for_step(self, step_name: str) -> pd.DataFrame:
        """
        Builds a pandas DataFrame from the stored XAI explanation results for a given
        step.
        Also saves attribution tensors to disk and logs them via MLflow.

        Args:
            step_name (str): Name of the processing step.

        Returns:
            pd.DataFrame: DataFrame containing the converted explanation results.

        Raises:
            ValueError: If no results exist for the given step.
        """
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

            # Save attribution tensor as .pt file if available
            if r.attribution is not None:
                torch.save(r.attribution, attribution_path)

            # Convert result to dict and handle tensor fields appropriately
            data = self._convert_result_to_csv_dict(r)
            data["attribution_path"] = attribution_path

            # Optionally log attribution file as MLflow artifact
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
        Converts an XAIExplanationResult instance into a CSV-friendly dictionary,
        converting or removing tensor fields as needed.

        Args:
            result (XAIExplanationResult): Explanation result to convert.

        Returns:
            Dict[str, Any]: Dictionary representation suitable for CSV output.
        """
        data = result.to_dict()

        tensor_fields = ["image", "attribution", "bbox", "bbox_info"]

        for field in tensor_fields:
            if field in data and data[field] is not None:
                if isinstance(data[field], torch.Tensor):
                    # Remove large tensors completely
                    if field in ["image", "attribution"]:
                        del data[field]
                    # Convert smaller tensors to string representation
                    elif field in ["bbox", "bbox_info"]:
                        data[field] = self._tensor_to_string_representation(data[field])

        # Special handling for explainer_result attribute
        if "explainer_result" in data and data["explainer_result"] is not None:
            data["explainer_result"] = self._convert_explainer_result_to_string(
                data["explainer_result"]
            )

        # Clean any remaining tensors from unknown fields
        data = self._clean_remaining_tensors(data)

        return data

    def _tensor_to_string_representation(self, tensor: torch.Tensor) -> str:
        """
        Converts a tensor to a concise string representation showing shape and type.

        Args:
            tensor (torch.Tensor): Tensor to convert.

        Returns:
            str: String summary of the tensor.
        """
        if tensor is None:
            raise ValueError("Tensor is None")

        try:
            if tensor.numel() <= 10:
                # Show full values for small tensors
                return f"<Tensor shape={tuple(tensor.shape)} values={tensor.tolist()}>"
            else:
                # Show summary info for large tensors
                return (
                    f"<Tensor shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"device={tensor.device}>"
                )
        except Exception as e:
            self._logger.warning(f"Error converting tensor to string: {e}")
            return f"<Tensor shape={tuple(tensor.shape)}>"

    def _convert_explainer_result_to_string(self, explainer_result: Any) -> str:
        """
        Converts an explainer result object to a summarized string representation,
        avoiding storing large tensors.

        Args:
            explainer_result: The explainer result instance.

        Returns:
            str: Summarized string representation of the explainer result.
        """
        if explainer_result is None:
            raise ValueError("explainer_result is None")

        try:
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
        """
        Recursively converts remaining tensors in a dictionary to string
        representations.

        Args:
            data (Dict[str, Any]): Dictionary possibly containing tensors.

        Returns:
            Dict[str, Any]: Cleaned dictionary with tensors converted to strings.
        """
        cleaned_data = {}

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = self._tensor_to_string_representation(value)
            elif isinstance(value, (list, tuple)):
                cleaned_data[key] = self._clean_tensor_list(value)
            elif isinstance(value, dict):
                cleaned_data[key] = self._clean_remaining_tensors(value)
            else:
                cleaned_data[key] = value

        return cleaned_data

    def _clean_tensor_list(self, items: Any) -> Any:
        """
        Recursively cleans tensors inside lists or tuples, converting them to string.

        Args:
            items (list or tuple): List or tuple potentially containing tensors.

        Returns:
            list or tuple: Cleaned list or tuple with tensors converted to strings.
        """
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
        evaluator: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Extends the DataFrame for a given step by adding evaluation metrics columns.

        Args:
            step_name (str): Name of the processing step.
            evaluation_summary (EvaluationSummary): Summary of evaluation metrics.
            individual_metrics (Optional[List[MetricResults]]): Precomputed individual
            metrics.
            evaluator (Optional[Any]): Evaluator object to compute individual metrics
            if not provided.

        Returns:
            pd.DataFrame: DataFrame extended with evaluation metric columns.
        """
        self._logger.info(
            f"Adding evaluation metrics to DataFrame for step '{step_name}'"
        )

        df = self.get_dataframe(step_name)

        if individual_metrics is not None:
            return self._add_individual_metrics_to_df(df, individual_metrics, step_name)

        elif evaluator is not None:
            return self._calculate_and_add_metrics_to_df(df, evaluator, step_name)

        else:
            return self._add_summary_metrics_to_df(df, evaluation_summary)

    def _add_individual_metrics_to_df(
        self, df: pd.DataFrame, individual_metrics: List[MetricResults], step_name: str
    ) -> pd.DataFrame:
        """
        Adds individual evaluation metrics to the DataFrame rows.

        Args:
            df (pd.DataFrame): DataFrame to update.
            individual_metrics (List[MetricResults]): List of metric results to add.
            step_name (str): Step name for logging and internal tracking.

        Returns:
            pd.DataFrame: Updated DataFrame with individual metrics.
        """
        self._logger.info(
            f"Adding {len(individual_metrics)} individual metrics to DataFrame"
        )

        if len(individual_metrics) != len(df):
            self._logger.warning(
                f"Mismatch: {len(individual_metrics)} metrics vs {len(df)} DataFrame "
                "rows. Attempting to match by image order."
            )

        # Initialize metric columns
        df["iou_score"] = None
        df["pixel_precision"] = None
        df["pixel_recall"] = None
        df["point_game_score"] = None
        df["evaluation_metrics_json"] = None

        import json

        # Populate metric columns
        for i, metrics in enumerate(individual_metrics):
            if i >= len(df):
                break

            if metrics and metrics.values:
                row_idx = i

                if "IoU" in metrics.values:
                    df.loc[row_idx, "iou_score"] = float(metrics.values["IoU"])

                if "PixelPrecisionRecall" in metrics.values:
                    ppr = metrics.values["PixelPrecisionRecall"]
                    if isinstance(ppr, dict):
                        df.loc[row_idx, "pixel_precision"] = float(
                            ppr.get("precision", 0)
                        )
                        df.loc[row_idx, "pixel_recall"] = float(ppr.get("recall", 0))
                    else:
                        df.loc[row_idx, "pixel_precision"] = float(ppr)

                if "point_game" in metrics.values:
                    df.loc[row_idx, "point_game_score"] = float(
                        metrics.values["point_game"]
                    )

                # Store all metrics as JSON string, converting tensors to items
                df.loc[row_idx, "evaluation_metrics_json"] = json.dumps(
                    {
                        k: (v if not isinstance(v, torch.Tensor) else v.item())
                        for k, v in metrics.values.items()
                    }
                )

        self.dataframes_per_step[step_name] = df
        self._logger.info("Individual metrics successfully added to DataFrame")
        return df

    def _calculate_and_add_metrics_to_df(
        self, df: pd.DataFrame, evaluator: Any, step_name: str
    ) -> pd.DataFrame:
        """
        Calculates individual metrics for stored results and adds them to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to update.
            evaluator (Any): Evaluator instance with an `evaluate_single_result` method.
            step_name (str): Step name for logging and internal tracking.

        Returns:
            pd.DataFrame: Updated DataFrame with calculated metrics.
        """
        self._logger.info("Calculating individual metrics for DataFrame rows")

        results = self.results_per_step.get(step_name, [])
        if not results:
            self._logger.warning(
                f"No results found for step '{step_name}' - cannot calculate metrics"
            )
            return df

        individual_metrics = []
        for result in results:
            metrics = evaluator.evaluate_single_result(result)
            individual_metrics.append(metrics)

        return self._add_individual_metrics_to_df(df, individual_metrics, step_name)

    def _add_summary_metrics_to_df(
        self, df: pd.DataFrame, evaluation_summary: EvaluationSummary
    ) -> pd.DataFrame:
        """
        Adds summary evaluation metrics (same values for all rows) to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to update.
            evaluation_summary (EvaluationSummary): Summary metrics to add.

        Returns:
            pd.DataFrame: Updated DataFrame with summary metrics.
        """
        self._logger.info("Adding summary metrics to all DataFrame rows")

        df["prediction_accuracy_summary"] = evaluation_summary.prediction_accuracy
        df["average_processing_time_summary"] = (
            evaluation_summary.average_processing_time
        )
        df["samples_with_bbox_summary"] = evaluation_summary.samples_with_bbox

        if evaluation_summary.metric_averages:
            for metric_name, avg_value in evaluation_summary.metric_averages.items():
                df[f"{metric_name}_summary"] = avg_value

        return df

    def save_dataframe_with_metrics(
        self,
        step_name: str,
        path: str,
        evaluation_summary: EvaluationSummary,
        individual_metrics: Optional[List[MetricResults]] = None,
        evaluator: Optional[Any] = None,
    ) -> None:
        """
        Saves the DataFrame with evaluation metrics as a CSV file.

        Args:
            step_name (str): Processing step name.
            path (str): File path to save the CSV.
            evaluation_summary (Optional[EvaluationSummary]): Optional summary metrics.
            individual_metrics (Optional[List[MetricResults]]): Optional precomputed
            metrics.
            evaluator (Optional[Any]): Optional evaluator to compute metrics if needed.
        """
        if evaluation_summary or individual_metrics or evaluator:
            df = self.add_evaluation_metrics_to_dataframe(
                step_name=step_name,
                evaluation_summary=evaluation_summary,
                individual_metrics=individual_metrics,
                evaluator=evaluator,
            )
        else:
            df = self.get_dataframe(step_name)

        df.to_csv(path, index=False)
        self._logger.info(f"DataFrame with metrics saved to {path}")

    def get_dataframe(self, step_name: str) -> pd.DataFrame:
        """
        Retrieves the DataFrame for a specific step, building it if necessary.

        Args:
            step_name (str): Step name.

        Returns:
            pd.DataFrame: DataFrame for the given step.
        """
        if step_name not in self.dataframes_per_step:
            return self.build_dataframe_for_step(step_name)
        return self.dataframes_per_step[step_name]

    def save_dataframe(self, step_name: str, path: str) -> None:
        """
        Saves the DataFrame for a step as a CSV file.

        Args:
            step_name (str): Step name.
            path (str): File path to save the CSV.
        """
        df = self.get_dataframe(step_name)
        df.to_csv(path, index=False)

    def reset(self) -> None:
        """
        Clears all stored results and DataFrames.
        """
        self.results_per_step.clear()
        self.dataframes_per_step.clear()

    def save_results(self, results: List[XAIExplanationResult], path: Path) -> None:
        """
        Saves a list of XAIExplanationResults to disk using torch.save.

        Args:
            results (List[XAIExplanationResult]): List of results to save.
            path (Path): File path to save the results.
        """
        torch.save(results, path)
