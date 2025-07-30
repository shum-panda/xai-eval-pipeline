import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from ressource_management.attribution_reference import AttributionReference
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
        Initialize the ResultManager.

        Args:
            attribution_dir: Directory path where attribution files will be stored
        """
        self._results: List[XAIExplanationResult] = []
        self._dataframe: Optional[pd.DataFrame] = None
        self._attribution_dir = attribution_dir
        self._logger = logging.getLogger(__name__)
        os.makedirs(self._attribution_dir, exist_ok=True)

    @property
    def results(self) -> List[XAIExplanationResult]:
        """Get the list of stored XAI results."""
        return self._results.copy()  # Return copy to prevent external modification

    @property
    def results_count(self) -> int:
        """Get the number of stored results."""
        return len(self._results)

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Get the DataFrame representation of results.

        Builds the DataFrame if it doesn't exist or is outdated.

        Returns:
            DataFrame containing all results with CSV-friendly format
        """
        if self._dataframe is None:
            return self.build_dataframe()
        return self._dataframe

    def add_results(self, new_results: List[XAIExplanationResult]) -> None:
        """
        Add new results to the manager.

        Args:
            new_results: List of XAI explanation results to add
        """
        self._results.extend(new_results)
        # Mark DataFrame as outdated so it gets rebuilt
        self._dataframe = None
        self._logger.debug(
            f"Added {len(new_results)} results. Total: {self.results_count}"
        )

    def build_dataframe(self) -> pd.DataFrame:
        """
        Build a DataFrame from all stored results.

        This method processes all results, saves their attributions locally,
        and creates a CSV-friendly DataFrame representation.

        Returns:
            DataFrame containing all results

        Raises:
            ValueError: If no results are available to build DataFrame
        """
        if not self._results:
            raise ValueError("No results available to build DataFrame.")

        records = []
        for result in self._results:
            # Convert to CSV-friendly dictionary
            data = self._convert_result_to_csv_dict(result)
            # Set attribution path from reference
            if isinstance(result.attribution, AttributionReference):
                data["attribution_path"] = str(result.attribution.file_path)
            elif result.attribution_path:
                data["attribution_path"] = result.attribution_path
            else:
                data["attribution_path"] = None
            records.append(data)

        df = pd.DataFrame(records)
        self._dataframe = df
        self._logger.info(f"Built DataFrame with {len(df)} rows")
        return df

    def _convert_result_to_csv_dict(
        self, result: XAIExplanationResult
    ) -> Dict[str, Any]:
        """
        Convert XAIExplanationResult to CSV-friendly dictionary.

        Args:
            result: XAI result to convert

        Returns:
            Dictionary with CSV-compatible data types
        """
        data = result.to_dict()

        # Remove large tensors
        for field in ["image", "attribution"]:
            data.pop(field, None)

        # Convert bbox to string if it's a tensor
        if "bbox" in data and isinstance(data["bbox"], torch.Tensor):
            data["bbox"] = self._tensor_to_string_representation(data["bbox"])

        # Convert explainer_result to summary string
        if "explainer_result" in data and data["explainer_result"] is not None:
            data["explainer_result"] = self._convert_explainer_result_to_string(
                data["explainer_result"]
            )

        # Clean any remaining unexpected tensors
        return self._clean_remaining_tensors(data)

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

    def save_dataframe(self, path: str) -> None:
        """
        Save the DataFrame as CSV.

        Args:
            path: File path where to save the CSV
        """
        df = self.dataframe
        df.to_csv(path, index=False)
        self._logger.info(f"DataFrame saved to {path}")

    def save_dataframe_with_metrics(
        self,
        path: str,
        evaluation_summary: Optional[EvaluationSummary] = None,
        individual_metrics: Optional[List[MetricResults]] = None,
    ) -> None:
        """
        Save DataFrame with evaluation metrics as CSV.

        Args:
            path: File path where to save the CSV
            evaluation_summary: Optional evaluation summary to add as columns
            individual_metrics: Optional pre-calculated individual metrics
        """
        # Extend DataFrame with metrics if available
        if evaluation_summary or individual_metrics:
            df = self.add_evaluation_metrics_to_dataframe(
                evaluation_summary=evaluation_summary,
                individual_metrics=individual_metrics,
            )
        else:
            df = self.dataframe

        # Save CSV
        df.to_csv(path, index=False)
        self._logger.info(f"DataFrame with metrics saved to {path}")

    def add_evaluation_metrics_to_dataframe(
        self,
        evaluation_summary: Optional[EvaluationSummary] = None,
        individual_metrics: Optional[List[MetricResults]] = None,
    ) -> pd.DataFrame:
        """
        Extend DataFrame with evaluation metrics.

        Args:
            evaluation_summary: Summary metrics to add to all rows
            individual_metrics: Pre-calculated metrics for each result

        Returns:
            DataFrame extended with metric columns
        """
        self._logger.info("Adding evaluation metrics to DataFrame")

        # Get current DataFrame
        df = self.dataframe

        # Method 1: Use pre-calculated individual metrics (preferred)
        if individual_metrics is not None:
            return self._add_individual_metrics_to_df(df, individual_metrics)

        # Method 2: Add only summary metrics
        elif evaluation_summary is not None:
            return self._add_summary_metrics_to_df(df, evaluation_summary)

        else:
            self._logger.warning("No metrics provided to add to DataFrame")
            return df

    def _add_individual_metrics_to_df(
        self, df: pd.DataFrame, individual_metrics: List[MetricResults]
    ) -> pd.DataFrame:
        """
        Add individual metrics to DataFrame dynamically.

        This method discovers all available metrics from the results and creates
        columns for them automatically, without hardcoding specific metric names.

        Args:
            df: DataFrame to extend
            individual_metrics: List of metrics for each result

        Returns:
            DataFrame with added metric columns
        """
        self._logger.info(
            f"Adding {len(individual_metrics)} individual metrics to DataFrame"
        )

        # Ensure we have the right number of metrics
        if len(individual_metrics) != len(df):
            self._logger.warning(
                f"Mismatch: {len(individual_metrics)} metrics vs {len(df)} DataFrame "
                "rows. "
                "Trying to match by image order."
            )

        # Discover all available metrics dynamically
        all_metric_keys = set()
        for metrics in individual_metrics:
            if metrics and metrics.values:
                all_metric_keys.update(metrics.values.keys())

        self._logger.info(f"Discovered metrics: {sorted(all_metric_keys)}")

        # Initialize columns for all discovered metrics + JSON column
        for metric_key in all_metric_keys:
            df[f"{metric_key}"] = None
        df["evaluation_metrics_json"] = None

        # Fill metric values dynamically
        for i, metrics in enumerate(individual_metrics):
            if i >= len(df):
                break

            if metrics and metrics.values:
                # Add all metrics as individual columns
                for metric_key, metric_value in metrics.values.items():
                    column_name = f"metric_{metric_key}"
                    try:
                        # Handle different metric value types
                        if isinstance(metric_value, dict):
                            # For nested dictionaries, flatten them
                            for sub_key, sub_value in metric_value.items():
                                nested_column = f"{column_name}_{sub_key}"
                                if nested_column not in df.columns:
                                    df[nested_column] = None
                                df.loc[i, nested_column] = (
                                    float(sub_value)
                                    if isinstance(sub_value, (int, float))
                                    else str(sub_value)
                                )
                        elif isinstance(metric_value, (int, float)):
                            df.loc[i, column_name] = float(metric_value)
                        elif isinstance(metric_value, torch.Tensor):
                            df.loc[i, column_name] = float(metric_value.item())
                        else:
                            df.loc[i, column_name] = str(metric_value)
                    except Exception as e:
                        self._logger.warning(f"Could not add metric {metric_key}: {e}")
                        df.loc[i, column_name] = str(metric_value)

                # Store all metrics as JSON for backup/debugging
                import json

                try:
                    df.loc[i, "evaluation_metrics_json"] = json.dumps(
                        {
                            k: (v if not isinstance(v, torch.Tensor) else v.item())
                            for k, v in metrics.values.items()
                        }
                    )
                except Exception as e:
                    self._logger.warning(f"Could not serialize metrics to JSON: {e}")
                    df.loc[i, "evaluation_metrics_json"] = str(metrics.values)

        # Update internal DataFrame
        self._dataframe = df
        self._logger.info(
            "Individual metrics successfully added to DataFrame "
            f"({len(all_metric_keys)} metric types)"
        )
        return df

    def _add_summary_metrics_to_df(
        self, df: pd.DataFrame, evaluation_summary: EvaluationSummary
    ) -> pd.DataFrame:
        """
        Add summary metrics to all DataFrame rows.

        Args:
            df: DataFrame to extend
            evaluation_summary: Summary containing aggregate metrics

        Returns:
            DataFrame with summary metric columns
        """
        self._logger.info("Adding summary metrics to all DataFrame rows")

        # Add summary metrics as new columns
        df["prediction_accuracy_summary"] = evaluation_summary.prediction_accuracy
        df["average_processing_time_summary"] = (
            evaluation_summary.average_processing_time
        )
        df["samples_with_bbox_summary"] = evaluation_summary.samples_with_bbox

        # Add average metric values
        if evaluation_summary.metric_averages:
            for metric_name, avg_value in evaluation_summary.metric_averages.items():
                df[f"{metric_name}_summary"] = avg_value

        # Update internal DataFrame
        self._dataframe = df
        return df

    def reset(self) -> None:
        """Reset all stored results and cached DataFrames."""
        self._results.clear()
        self._dataframe = None
        self._logger.info("ResultManager reset")

    def get_latest_results(self, n: int = 10) -> List[XAIExplanationResult]:
        """
        Get the latest n results.

        Args:
            n: Number of latest results to return

        Returns:
            List of the latest n results
        """
        return self._results[-n:] if len(self._results) >= n else self._results
