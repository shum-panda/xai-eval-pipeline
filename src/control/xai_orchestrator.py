import dataclasses
import logging
import time
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import mlflow
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
from tqdm import tqdm

import src.pipeline_moduls.evaluation.metrics  # noqa: F401
from src.control.utils.config_dataclasses.master_config import MasterConfig
from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.control.utils.error.xai_explanation_error import XAIExplanationError
from src.control.utils.set_up_logger import setup_logger
from src.control.utils.with_cuda_cleanup import with_cuda_cleanup
from src.pipeline_moduls.data.dataclass.xai_input_batch import XAIInputBatch
from src.pipeline_moduls.data.image_net_label_mapper import ImageNetLabelMapper
from src.pipeline_moduls.data.image_net_val_dataset import (
    ImageNetValDataset,
    create_dataloader,
)
from src.pipeline_moduls.data.utils.collate_fn import explain_collate_fn
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.xai_evaluator import XAIEvaluator
from src.pipeline_moduls.metaanlyse.xai_meta_analysis import XaiMetaAnalysis
from src.pipeline_moduls.models.base.interface.xai_model import XAIModel
from src.pipeline_moduls.models.base.xai_model_factory import XAIModelFactory
from src.pipeline_moduls.resultmanager.result_manager import ResultManager
from src.pipeline_moduls.visualization.visualisation import Visualiser
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.xai_factory import XAIFactory
from src.ressource_management.attribution_reference import AttributionReference


class XAIOrchestrator:
    """
    Central orchestrator of the entire XAI pipeline.
    Coordinates models, datasets, explainers, evaluation, and visualization,
    but does not perform evaluation itself.
    """

    def __init__(self, config: MasterConfig) -> None:
        """
        Initialize the XAI Orchestrator with the given configuration.

        Args:
            config (MasterConfig): Configuration object loaded via Hydra.
        """

        project_root = Path(__file__).resolve().parents[2]
        self._pipeline_status: str = "initialized"
        self._current_step: str = "none"
        self._pipeline_error: Optional[Exception] = None
        self._individual_metrics: Optional[List[Any]] = None
        self._mlflow_run: Optional[mlflow.ActiveRun] = None
        self._logger: Logger = logging.getLogger(__name__)
        self._config: MasterConfig = config
        self._result_manager: ResultManager = ResultManager()

        # Pfade relativ zum Projekt-Root
        mapping_file = project_root / "data" / "raw" / "final_mapping.txt"
        imagenet_class_index_file = (
            project_root / "data" / "raw" / "imagenet_class_index.json"
        )

        self.label_mapper = ImageNetLabelMapper(
            mapping_file=mapping_file,
            imagenet_class_index_file=imagenet_class_index_file,
        )

        # Setup Logger
        try:
            setup_logger(self._config.logging)
            self._logger.info(f"Starting logging in {self._config.logging.level} level")
        except ValueError as e:
            raise e

        # Setup factories
        self._model_factory: XAIModelFactory = XAIModelFactory()
        self._xai_factory: XAIFactory = XAIFactory()

        # Load model
        self._model: XAIModel = self._model_factory.create(config.model.name)

        # Evaluator and Visualizer
        self._logger.debug(config.metric.kwargs)
        self._evaluator: XAIEvaluator = XAIEvaluator(metric_kwargs=config.metric.kwargs)
        self._visualiser: Visualiser = Visualiser(
            show=self._config.visualization.show,
            save_path=Path(self._config.experiment.output_dir),
        )

        self._streaming_metrics = []  # Store individual metrics during streaming
        self._streaming_stats = {}  # Running statistics
        self._evaluation_start_time = None
        self._total_evaluation_time = 0.0

        self._logger.info("Orchestrator initialized:")
        self._logger.info(f"  Model: {config.model.name} on {self._model.device}")
        self._logger.info(
            f"  Available explainers: {self._xai_factory.list_available_explainers()}"
        )

    def _reset_streaming_state(self) -> None:
        """Initialize streaming evaluation state"""
        self._streaming_metrics = []
        self._evaluation_start_time = time.time()
        self._total_evaluation_time = 0.0
        self._streaming_stats = {
            "total_samples": 0,
            "correct_predictions": 0,
            "samples_with_bbox": 0,
            "total_processing_time": 0.0,
            "metric_sums": {},
            "metric_counts": {},
        }
        self._logger.info("Streaming evaluation state initialized")

    @property
    def pipeline_status(self) -> Dict[str, Union[str, bool, None]]:
        """
        Returns current pipeline status useful for monitoring or debugging.

        Returns:
            Dict[str, Union[str, bool, None]]: Status information dictionary.
        """
        return {
            "status": self._pipeline_status,
            "current_step": self._current_step,
            "has_error": self._pipeline_error is not None,
            "error_details": (
                str(self._pipeline_error) if self._pipeline_error else None
            ),
            "mlflow_active": self._mlflow_run is not None,
        }

    def reset_pipeline_state(self) -> None:
        """
        Resets pipeline status to initial state for new runs.
        """
        self._pipeline_status = "initialized"
        self._current_step = "none"
        self._pipeline_error = None
        self._logger.info("Pipeline state reset")

    def run(self) -> Dict[str, Any]:
        """
        Extended run() method with status tracking.
        (Robust error handling comes in step 3)

        Returns:
            Dict[str, Any]: Summary information of the run on success.

        Raises:
            Exception: Propagates exceptions encountered during execution.
        """
        self._pipeline_status = "running"

        try:
            self._current_step = "experiment_preparation"
            self._logger.info(f"Starting step: {self._current_step}")
            self.prepare_experiment()

            self._current_step = "dataloader_setup"
            self._logger.info(f"Starting step: {self._current_step}")
            dataloader = self.setup_dataloader(
                project_root=None,
                batch_size=self._config.data.batch_size,
                num_workers=self._config.data.num_workers,
                pin_memory=self._config.data.pin_memory,
                shuffle=self._config.data.shuffle,
                target_size=self._config.data.resize,
                transform=None,
            )

            self._current_step = "explainer_creation"
            self._logger.info(f"Starting step: {self._current_step}")
            explainer = self.create_explainer(
                explainer_name=self._config.xai.name, **self._config.xai.kwargs
            )

            self._current_step = "pipeline_execution"
            self._logger.info(f"Starting step: {self._current_step}")
            xai_results = self.run_pipeline(dataloader, explainer)
            self._logger.info(
                "XAI result example after transformation:" f" {xai_results[0]}"
            )
            self._current_step = "results_evaluation"
            self._logger.info(f"Starting step: {self._current_step}")
            summary = self._create_evaluation_summary_from_streaming()

            self._current_step = "results_saving"
            self._logger.info(f"Starting step: {self._current_step}")
            self.save_results(summary)

            self._current_step = "meta analysis"
            self._logger.info(f"Starting step: {self._current_step}")
            self.xai_meta_analyse()

            self._current_step = "visualization"
            self._logger.info(f"Starting step: {self._current_step}")
            self.visualize_results_if_needed(xai_results, summary)
        except Exception as e:
            self._pipeline_status = "failed"
            self._pipeline_error = e
            self._logger.error(f"PIPELINE FAILED at step: {self._current_step}")
            self._logger.error(f"Error: {str(e)}")
            if self._config.experiment.emergency_export_enabled:
                self.emergency_export()
            raise
        else:
            self._logger.info("Pipeline completed successfully!")
            return {
                "status": "success",
                "total_samples": len(xai_results),
                "output_dir": self._config.experiment.output_dir,
                "explainer": self._config.xai.name,
                "_model": self._config.model.name,
            }
        finally:
            self._current_step = "finalization"
            self._logger.info(f"Starting step: {self._current_step}")
            self.finalize_run()

            self._pipeline_status = "completed"
            self._current_step = "completed"

    def emergency_export(self) -> None:
        """
        Perform emergency export of current results in case of pipeline failure.

        This method attempts to save whatever results have been processed so far
        to prevent complete data loss in case of crashes or interruptions.
        """
        try:
            if self._result_manager.results_count > 0:
                emergency_path = (
                    Path(self._config.experiment.output_dir) / "emergency_results.csv"
                )
                self._result_manager.save_dataframe(str(emergency_path))
                self._logger.info(f"Emergency export completed: {emergency_path}")
                self._logger.info(
                    f"Exported {self._result_manager.results_count} results"
                )
        except Exception as e:
            self._logger.error(f"Emergency export failed: {e}")

    def prepare_experiment(self) -> None:
        """
        Starts the MLflow experiment and logs parameters.
        """
        self._logger.info(f"Starting experiment: {self._config.experiment.name}")
        if mlflow.active_run() is None:
            self._mlflow_run = mlflow.start_run(run_name=self._config.experiment.name)
        else:
            self._mlflow_run = mlflow.active_run()

        mlflow.log_param("_model_name", self._config.model.name)
        mlflow.log_param("explainer_name", self._config.xai.name)
        mlflow.log_param("batch_size", self._config.data.batch_size)
        mlflow.log_param("max_batches", self._config.data.max_batches)

    def run_pipeline(
        self,
        dataloader: DataLoader[ImageNetValDataset],
        explainer: BaseExplainer,
    ) -> List[XAIExplanationResult]:
        """
        Runs the pipeline by processing batches from the dataloader through the
        explainer.

        Each explanation result is immediately transformed (if enabled) and stored.

        Args:
            dataloader: DataLoader providing batches of data to explain.
            explainer: Explainer instance used to generate explanations.

        Returns:
            List of transformed explanation results for all processed samples.
        """
        self._logger.info(f"Processing samples with {self._config.xai.name}...")

        results: List[XAIExplanationResult] = []
        self._reset_streaming_state()

        label_map = self.label_mapper.class_id_to_label
        class_to_val_tensor = self.label_mapper.class_to_val_tensor

        max_batches = self._config.data.max_batches
        estimated_total = (
            len(dataloader.dataset)
            if max_batches is None
            else max_batches * dataloader.batch_size
        )

        for batch in tqdm(
                self.process_dataloader(
                    dataloader=dataloader,
                    explainer=explainer,
                    max_batches=max_batches,
                ),
                desc="Running XAI pipeline",
                total=estimated_total,
        ):
            transformed_batch = [
                self.transform_result(
                    result=r,
                    transform=self._config.model.transform,
                    class_to_val_tensor=class_to_val_tensor,
                    label_lookup=label_map,
                )
                for r in batch
            ]
            for transformed_result in transformed_batch:
                if transformed_result.attribution is not None:
                    self._evaluate_single_result_streaming(transformed_result)

            results.extend(transformed_batch)
            self._result_manager.add_results(transformed_batch)
        self._individual_metrics = self._streaming_metrics
        self._logger.info(
            f"Finished processing. Total results collected: {len(results)}"
        )
        return results

    def _create_evaluation_summary_from_streaming(self) -> EvaluationSummary:
        """Create evaluation summary from streaming statistics"""
        stats = self._streaming_stats

        # Calculate averages
        prediction_accuracy = (
            stats["correct_predictions"] / stats["total_samples"]
            if stats["total_samples"] > 0
            else 0.0
        )

        average_processing_time = (
            stats["total_processing_time"] / stats["total_samples"]
            if stats["total_samples"] > 0
            else 0.0
        )

        # Calculate metric averages
        metric_averages = {}
        for metric_name, total_sum in stats["metric_sums"].items():
            count = stats["metric_counts"][metric_name]
            if count > 0:
                metric_averages[f"average_{metric_name}"] = total_sum / count

        # Add timing metrics
        metric_averages["total_evaluation_time"] = self._total_evaluation_time
        metric_averages["average_evaluation_time_per_sample"] = (
            self._total_evaluation_time / stats["total_samples"]
            if stats["total_samples"] > 0
            else 0.0
        )

        # Create summary with ALL required fields

        summary = EvaluationSummary(
            explainer_name=self._config.xai.name,
            model_name=self._config.model.name,
            total_samples=stats["total_samples"],
            samples_with_bbox=stats["samples_with_bbox"],
            prediction_accuracy=prediction_accuracy,
            correct_predictions=stats["correct_predictions"],
            average_processing_time=average_processing_time,
            total_processing_time=stats["total_processing_time"],
            evaluation_timestamp=datetime.now().isoformat(),
            metric_averages=metric_averages,
        )

        return summary

    def _evaluate_single_result_streaming(self, result: XAIExplanationResult):
        """Evaluate single result and store metrics"""
        eval_start = time.time()

        # Dein bestehender Evaluator
        metrics = self._evaluator.evaluate_single_result(result)

        eval_end = time.time()
        self._total_evaluation_time += eval_end - eval_start

        # Store für später
        self._streaming_metrics.append(metrics)

        # Update stats
        self._update_streaming_stats(result, metrics)
        return metrics

    @with_cuda_cleanup
    def _save_attribution_and_create_reference(
        self,
        attribution: torch.Tensor,
        model_name: str,
        explainer_name: str,
        image_name: str,
    ) -> Path:
        """
        Save attribution tensor to disk and return file path.
        """
        attribution_dir = (
            Path(self._config.experiment.output_dir)
            / "attributions"
            / model_name
            / explainer_name
        )
        attribution_dir.mkdir(parents=True, exist_ok=True)

        attribution_path = attribution_dir / f"{image_name}_attribution.pt"

        # Save tensor to disk
        torch.save(attribution, attribution_path)
        self._logger.debug(f"Saved attribution to {attribution_path}")

        return attribution_path

    def _update_streaming_stats(self, result: XAIExplanationResult, metrics) -> None:
        """Update running statistics"""
        stats = self._streaming_stats

        # Basic stats
        stats["total_samples"] += 1
        if result.prediction_correct:
            stats["correct_predictions"] += 1
        if result.has_bbox:
            stats["samples_with_bbox"] += 1
        stats["total_processing_time"] += result.processing_time

        # Metric stats
        if metrics and metrics.values:
            for metric_name, metric_value in metrics.values.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in stats["metric_sums"]:
                        stats["metric_sums"][metric_name] = 0.0
                        stats["metric_counts"][metric_name] = 0

                    stats["metric_sums"][metric_name] += float(metric_value)
                    stats["metric_counts"][metric_name] += 1

                elif isinstance(metric_value, dict):
                    # Handle nested metrics
                    for sub_key, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            full_key = f"{metric_name}_{sub_key}"
                            if full_key not in stats["metric_sums"]:
                                stats["metric_sums"][full_key] = 0.0
                                stats["metric_counts"][full_key] = 0

                            stats["metric_sums"][full_key] += float(sub_value)
                            stats["metric_counts"][full_key] += 1



    def save_results(self, summary: EvaluationSummary) -> None:
        """
        Save results and evaluation summary to disk and log artifacts to MLflow.

        Args:
            summary: Evaluation summary object containing aggregate metrics
        """
        output_dir = Path(self._config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV with metrics using PRE-CALCULATED individual metrics
        csv_path = output_dir / "results_with_metrics.csv"

        self._logger.info("Using pre-calculated individual metrics for CSV export")
        self._result_manager.save_dataframe_with_metrics(
            path=str(csv_path),
            evaluation_summary=summary,
            individual_metrics=self._streaming_metrics,
        )

        mlflow.log_artifact(str(csv_path), artifact_path="evaluation/csv_results")
        self._logger.info(f"CSV with metrics saved to {csv_path}")

        # Save summary as YAML
        summary_path = output_dir / "metrics_summary.yaml"
        with open(summary_path, "w") as f:
            import yaml

            yaml.dump(dataclasses.asdict(summary), f)
        mlflow.log_artifact(
            str(summary_path), artifact_path="evaluation/metrics_summary"
        )
        self._logger.info(f"Metrics summary saved to {summary_path}")

    def visualize_results_if_needed(
        self, results: List[XAIExplanationResult], summary: EvaluationSummary
    ) -> None:
        """
        Generates and logs visualizations if configured to do so.

        Args:
            results (List[XAIExplanationResult]): List of explanation results.
            summary (EvaluationSummary): Evaluation summary.
        """
        if not self._config.visualization.save and not self._config.visualization.show:
            return

        output_dir = Path(self._config.experiment.output_dir)
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        visualizer = Visualiser(show=self._config.visualization.show, save_path=vis_dir)

        max_vis = min(self._config.visualization.max_visualizations, len(results))
        self._logger.info(f"Creating {max_vis} visualizations...")

        # Use individual metrics if available
        individual_metrics = getattr(self, "_individual_metrics", None)

        for i in range(max_vis):
            result = results[i]

            if individual_metrics and i < len(individual_metrics):
                metrics_for_this_result = individual_metrics[i]
                self._logger.debug(f"Using individual metrics for {result.image_name}")
            else:
                # Fallback to summary (not ideal)
                metrics_for_this_result = summary
                self._logger.warning(f"Using summary metrics for {result.image_name}")

            vis_path = visualizer.create_visualization(result, metrics_for_this_result)

            if vis_path and self._config.visualization.save:
                mlflow.log_artifact(vis_path, artifact_path="visualizations")

        self._logger.info(f"Generated {max_vis} visualizations in {vis_dir}")

    def finalize_run(self) -> None:
        """
        Ends the MLflow run if active and logs the run ID.
        """
        if self._mlflow_run and mlflow.active_run() is not None:
            mlflow.end_run()
            self._logger.info(f"MLflow run ended: {self._mlflow_run.info.run_id}")
            self._mlflow_run = None

    def setup_dataloader(
        self,
        project_root: Optional[Path],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        target_size: Optional[List[int]],
        transform: Optional[transforms.Compose],
    ) -> DataLoader[ImageNetValDataset]:
        """
        Sets up the ImageNet DataLoader with configurable parameters.

        Args:
            project_root (Optional[Path]): Root path of the project.
                If None, it is inferred automatically.
            batch_size (int): Batch size for loading the dataset.
            num_workers (int): Number of subprocesses used for data loading.
            pin_memory (bool): Whether to pin memory (recommended for GPU).
            shuffle (bool): Whether to shuffle the dataset during loading.
            target_size (Optional[List[int]]): List with to Arguments for Target size
            for resizing images.
            transform (Optional[transforms.Compose]): Optional custom transform to
            apply.

        Returns:
            DataLoader: Configured PyTorch DataLoader with ImageNet validation data.

        Notes:
            Uses `create_dataloader()` internally and expects compatibility with
            `ImageNetSample` and `explain_collate_fn`.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        image_dir = project_root / "data" / "extracted" / "validation_images"
        annot_dir = project_root / "data" / "extracted" / "bounding_boxes"
        label_file = (
            project_root / "data" / "raw" / "ILSVRC2012_validation_ground_truth.txt"
        )

        dataloader = create_dataloader(
            image_dir=image_dir,
            annot_dir=annot_dir,
            label_file=label_file,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            target_size=target_size,
            transform=transform,
            custom_collate_fn=explain_collate_fn,
        )

        self._logger.info(
            f"DataLoader setup complete: {len(dataloader.dataset)} samples in "  # type: ignore
            f"{len(dataloader)} batches "
            f"(batch_size={batch_size}, shuffle={shuffle})"
        )

        return dataloader

    def create_explainer(
        self, explainer_name: str, **additional_kwargs: Any
    ) -> BaseExplainer:
        """
        Creates an explainer with runtime parameter validation.

        Args:
            explainer_name (str): Name of the explainer.
            **additional_kwargs: Additional parameters to override config.

        Returns:
            BaseExplainer: Configured explainer instance.

        Raises:
            TypeError, ValueError, Exception: If creation fails.
        """
        logger = logging.getLogger(__name__)

        config_kwargs = self._config.xai.kwargs.copy()
        config_kwargs.update(additional_kwargs)

        use_defaults = self._config.xai.use_defaults

        logger.info(f"Creating {explainer_name} explainer...")
        logger.info(f"Use defaults: {use_defaults}")
        logger.debug(f"Parameters: {config_kwargs}")

        try:
            explainer = self._xai_factory.create_explainer(
                name=explainer_name,
                model=self._model,
                use_defaults=use_defaults,
                **config_kwargs,
            )
            return explainer
        except TypeError as e:
            logger.error(f"Failed to create {explainer_name}: {e}")
            logger.error("Check if the parameters have the right types")
            logger.error("Check your config parameters or set 'use_defaults: true'")
            raise
        except ValueError as e:
            logger.error(f"Failed to create {explainer_name}: {e}")
            logger.error("Check your config parameters or set 'use_defaults: true'")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating {explainer_name}: {e}")
            raise

    @with_cuda_cleanup
    def process_dataloader(
        self,
        dataloader: DataLoader[ImageNetValDataset],
        explainer: BaseExplainer,
        max_batches: Optional[int] = None,
    ) -> Iterator[List[XAIExplanationResult]]:
        """
        Iterates over a DataLoader and explains each batch using a given XAI explainer.
        Yields XAIExplanationResult objects for each image in a memory-efficient way.

        Args:
            dataloader (DataLoader): DataLoader yielding batches via custom collate_fn.
            explainer (BaseExplainer): XAI explainer instance.
            max_batches (Optional[int]): Limit on number of batches to process.

        Yields:
            XAIExplanationResult: One result per image.
        """
        total_batches = (
            min(len(dataloader), max_batches) if max_batches else len(dataloader)
        )
        self._logger.info(f"Starting processing of {total_batches} batches...")

        failed_batches: List[int] = []

        # tqdm Progressbar einrichten
        with tqdm(total=total_batches, desc="Explaining batches") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                # Custom collate function ensures batch is XAIInputBatch
                xai_batch: XAIInputBatch = batch  # type: ignore
                try:
                    results = self.explain_batch(xai_batch, explainer)
                except XAIExplanationError as e:
                    failed_batches.append(batch_idx)
                    image_names = (
                        xai_batch.image_names
                        if xai_batch.image_names
                        else ["<unknown>"] * len(xai_batch.images_tensor)
                    )
                    self._logger.warning(
                        f"[Batch {batch_idx}] could not be explained: {e}"
                    )
                    self._logger.debug(
                        f"[Batch {batch_idx}] image names: {image_names}"
                    )
                    pbar.update(1)
                    continue
                except TypeError as e:
                    failed_batches.append(batch_idx)
                    self._logger.error(f"[Batch {batch_idx}] TypeError: {e}")
                    self._logger.debug(
                        "Batch content: " f"{[path for path in xai_batch.image_paths]}"
                    )
                    pbar.update(1)
                    continue
                except Exception as e:
                    failed_batches.append(batch_idx)
                    self._logger.error(f"[Batch {batch_idx}] Unexpected error: {e}")
                    self._logger.debug(
                        "Batch content: " f"{[path for path in xai_batch.image_paths]}"
                    )
                    pbar.update(1)
                    continue

                yield results

                pbar.update(1)  # Fortschritt nach jedem Batch aktualisieren

        if failed_batches:
            self._logger.warning(
                f"Processing finished with {len(failed_batches)} failed batches: "
                f"{failed_batches}"
            )

        if max_batches is not None and len(failed_batches) == max_batches:
            raise XAIExplanationError("No valid results from any batch")

    @with_cuda_cleanup
    def explain_batch(
        self, batch: XAIInputBatch, explainer: BaseExplainer
    ) -> List[XAIExplanationResult]:
        """
        Explains a batch of images using the provided XAI explainer.

        Args:
            batch (XAIInputBatch): Batch of images and metadata.
            explainer (BaseExplainer): Explainer instance.

        Returns:
            List[XAIExplanationResult]: List of explanation results for the batch.

        Raises:
            XAIExplanationError: If explanation fails.
        """
        start_time = time.time()
        current_device = self._model.device
        images = batch.images_tensor.to(current_device)
        labels_tensor = batch.labels_tensor.to(current_device)

        try:
            explanation_result = explainer.explain(
                images, labels_tensor, self._config.experiment.top_k
            )
            attributions = explanation_result.attributions
            pred_classes = explanation_result.predictions
            confidence = explanation_result.confidence
            target_labels = explanation_result.target_labels
            topk_preds = explanation_result.topk_predictions
            topk_confs = explanation_result.topk_confidences

        except Exception as e:
            self._logger.error(f"Error explaining batch: {e}")
            raise

        processing_time = time.time() - start_time
        results: List[XAIExplanationResult] = []

        for i in range(images.size(0)):
            if pred_classes.dim() == 2:
                predicted_class = pred_classes[i].argmax().item()
            else:
                predicted_class = pred_classes[i].item()

            true_label = int(target_labels[i])
            image_name = batch.image_names[i] if batch.image_names else f"image_{i}"
            top_k_predictions = topk_preds[i].tolist() if topk_preds is not None else []
            top_k_confidences = topk_confs[i].tolist() if topk_confs is not None else []
            attribution_path = self._save_attribution_and_create_reference(
                attributions[i],
                self._config.model.name,
                self._config.xai.name,
                image_name,
            )
            attribution = AttributionReference(attribution_path)
            result = XAIExplanationResult(
                image_path=batch.image_paths[i],
                image_name=image_name,
                predicted_class=int(predicted_class),
                prediction_confidence=float(confidence[i].item()),
                true_label=true_label,
                prediction_correct=(predicted_class == true_label),
                topk_predictions=top_k_predictions,
                topk_confidences=top_k_confidences,
                attribution=attribution,
                attribution_path=str(attribution_path),
                explainer_result=explanation_result,
                explainer_name=explainer.get_name(),
                explainer_params=explainer.parameters,
                has_bbox=batch.boxes_list[i].numel() > 0,
                bbox=batch.boxes_list[i].detach().cpu(),
                model_name=self._config.model.name,
                processing_time=processing_time / images.size(0),
                timestamp=f"{start_time}",
            )
            results.append(result)

        return results
    def transform_result(
        self,
        result: XAIExplanationResult,
        transform: bool,
        class_to_val_tensor: torch.Tensor,
        label_lookup: Dict[int, str],
    ) -> XAIExplanationResult:
        orig_class = result.predicted_class
        mapped_class = (
            int(class_to_val_tensor[orig_class] - 1) if transform else (orig_class)
        )

        true_label_val_idx = (
            result.true_label if result.true_label is not None else None
        )
        true_label_name = (
            label_lookup.get(true_label_val_idx, f"Class {true_label_val_idx}")
            if true_label_val_idx is not None
            else None
        )

        prediction_correct = (
            (mapped_class == true_label_val_idx)
            if true_label_val_idx is not None
            else False
        )

        return dataclasses.replace(
            result,
            predicted_class=mapped_class,
            predicted_class_before_transform=(
                orig_class if transform else result.predicted_class_before_transform
            ),
            predicted_class_name=label_lookup.get(
                mapped_class, f"Class {mapped_class}"
            ),
            true_label_name=true_label_name,
            prediction_correct=prediction_correct,
        )

    def xai_meta_analyse(self):
        """
        Run meta-analysis and store results locally and in MLflow.
        Results are written to:
            - results/<experiment>/meta_analysis/plots/
            - results/<experiment>/meta_analysis/threshold_iou_score.csv
        """
        output_dir = Path(self._config.experiment.output_dir)
        csv_path = output_dir / "results_with_metrics.csv"

        if not csv_path.exists():
            self._logger.error(f"Expected CSV not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)
        analysis = XaiMetaAnalysis(df)

        meta_dir = output_dir / "meta_analysis"
        plot_dir = meta_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Korrelationen berechnen & loggen
        corrs = analysis.correlation_with_correctness()
        mlflow.log_metrics({f"corr_{k}": v for k, v in corrs.items()})

        # Boxplots generieren und speichern
        plots = analysis.plot_metric_vs_correctness()
        for metric_name, fig in plots.items():
            filename = f"{metric_name}_vs_correctness.png"
            save_path = plot_dir / filename
            fig.savefig(save_path)
            plt.close(fig)
            mlflow.log_artifact(str(save_path), artifact_path="meta_analysis/plots")

        # Threshold-Analyse
        grouped, threshold_fig = analysis.threshold_analysis("metric_IoU")
        threshold_img_path = plot_dir / "threshold_iou_score.png"
        threshold_fig.savefig(threshold_img_path)
        plt.close(threshold_fig)
        mlflow.log_artifact(str(threshold_img_path),
                            artifact_path="meta_analysis/plots")

        # Gruppierte Daten als CSV speichern & loggen
        threshold_csv_path = meta_dir / "threshold_iou_score.csv"
        grouped.to_csv(threshold_csv_path)
        mlflow.log_artifact(str(threshold_csv_path), artifact_path="meta_analysis")

        self._logger.info(f"Meta-Analyse gespeichert in {meta_dir.resolve()}")
