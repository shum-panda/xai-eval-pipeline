import dataclasses
import logging
import math
import time
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import mlflow
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore

import src.pipeline_moduls.evaluation.metrics  # noqa: F401
from src.control.utils.config_dataclasses.master_config import MasterConfig
from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.control.utils.error.xai_explanation_error import XAIExplanationError
from src.control.utils.set_up_logger import setup_logger
from src.control.utils.with_cuda_cleanup import with_cuda_cleanup
from src.pipeline_moduls.data.dataclass.xai_input_batch import XAIInputBatch
from src.pipeline_moduls.data.image_net_val_dataset import (
    ImageNetValDataset,
    create_dataloader,
)
from src.pipeline_moduls.data.utils.collate_fn import explain_collate_fn
from src.pipeline_moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipeline_moduls.evaluation.xai_evaluator import XAIEvaluator
from src.pipeline_moduls.models.base.interface.xai_model import XAIModel
from src.pipeline_moduls.models.base.xai_model_factory import XAIModelFactory
from src.pipeline_moduls.ResultManager.result_manager import ResultManager
from src.pipeline_moduls.visualization.visualisation import Visualiser
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.xai_factory import XAIFactory


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
        self._pipeline_status: str = "initialized"
        self._current_step: str = "none"
        self._pipeline_error: Optional[Exception] = None
        self._individual_metrics: Optional[List[Any]] = None
        self._mlflow_run: Optional[mlflow.ActiveRun] = None
        self._logger: Logger = logging.getLogger(__name__)
        self._config: MasterConfig = config
        self._result_manager: ResultManager = ResultManager()

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

        self._logger.info("Orchestrator initialized:")
        self._logger.info(f"  Model: {config.model.name} on {self._model.device}")
        self._logger.info(
            f"  Available explainers: {self._xai_factory.list_available_explainers()}"
        )

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
            results = self.run_pipeline(dataloader, explainer)

            self._current_step = "results_evaluation"
            self._logger.info(f"Starting step: {self._current_step}")
            summary = self.evaluate_results(results)

            self._current_step = "results_saving"
            self._logger.info(f"Starting step: {self._current_step}")
            self.save_results(results, summary)

            self._current_step = "visualization"
            self._logger.info(f"Starting step: {self._current_step}")
            self.visualize_results_if_needed(results, summary)

            self._current_step = "finalization"
            self._logger.info(f"Starting step: {self._current_step}")
            self.finalize_run()

            self._pipeline_status = "completed"
            self._current_step = "completed"
            self._logger.info("Pipeline completed successfully!")

            return {
                "status": "success",
                "total_samples": len(results),
                "output_dir": self._config.experiment.output_dir,
                "explainer": self._config.xai.name,
                "model": self._config.model.name,
            }

        except Exception as e:
            self._pipeline_status = "failed"
            self._pipeline_error = e
            self._logger.error(f"PIPELINE FAILED at step: {self._current_step}")
            self._logger.error(f"Error: {str(e)}")
            raise

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
        Runs the pipeline: process DataLoader batches through the explainer.

        Args:
            dataloader (DataLoader): DataLoader providing batches to explain.
            explainer (BaseExplainer): The explainer instance to generate explanations.

        Returns:
            List[XAIExplanationResult]: List of explanation results for all images.
        """
        self._logger.info(f"Processing samples with {self._config.xai.name}...")
        results: List[XAIExplanationResult] = []

        for result in self.process_dataloader(
            dataloader=dataloader,
            explainer=explainer,
            max_batches=self._config.data.max_batches,
        ):
            results.append(result)
            self._result_manager.add_results("step_1", [result])

        self._logger.info(
            "Finished processing. Total results collected: " f"{len(results)}"
        )
        return results

    def evaluate_results(
        self, results: List[XAIExplanationResult]
    ) -> EvaluationSummary:
        """
        Evaluates a batch of explanation results and logs metrics to MLflow.
        """
        self._logger.info("Calculating evaluation metrics...")

        individual_metrics: List[Any] = []
        correct_predictions = 0
        total_processing_time = 0.0

        self._logger.info(
            f"Processing {len(results)} results for individual metrics..."
        )

        for i, result in enumerate(results):
            if result.prediction_correct is not None and result.prediction_correct:
                correct_predictions += 1

            total_processing_time += result.processing_time

            metrics = self._evaluator.evaluate_single_result(result)
            individual_metrics.append(metrics)

            if (i + 1) % 10 == 0:
                self._logger.info(
                    f"Processed {i + 1}/{len(results)} individual metrics"
                )

        self._individual_metrics = individual_metrics

        summary = self._evaluator.create_summary_from_individual_metrics(
            results=results,
            individual_metrics=individual_metrics,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

        self._logger.info("Evaluation metrics calculation finished!")

        for key, value in summary.to_dict().items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        return summary

    def save_results(
        self, results: List[XAIExplanationResult], summary: EvaluationSummary
    ) -> None:
        """
        Saves results and evaluation summary to disk and logs artifacts to MLflow.

        Args:
            results (List[XAIExplanationResult]): List of explanation results.
            summary (EvaluationSummary): Evaluation summary object.
        """
        output_dir = Path(self._config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "results.pt"
        self._result_manager.save_results(results, results_path)
        mlflow.log_artifact(str(results_path), artifact_path="evaluation/results")
        self._logger.info(f"Serialized results saved to {results_path}")

        csv_path = output_dir / "results_with_metrics.csv"

        individual_metrics = getattr(self, "_individual_metrics", None)

        if individual_metrics:
            self._logger.info("Using pre-calculated individual metrics for CSV export")
            self._result_manager.save_dataframe_with_metrics(
                step_name="step_1",
                path=str(csv_path),
                evaluation_summary=summary,
                individual_metrics=individual_metrics,
            )
        else:
            self._logger.warning(
                "No individual metrics found, creating CSV without detailed metrics"
            )
            self._result_manager.save_dataframe_with_metrics(
                step_name="step_1", path=str(csv_path), evaluation_summary=summary
            )

        mlflow.log_artifact(str(csv_path), artifact_path="evaluation/csv_results")
        self._logger.info(f"CSV with metrics saved to {csv_path}")

        # Save summary
        summary_path = output_dir / "metrics_summary.yaml"
        with open(summary_path, "w") as f:
            import yaml

            yaml.dump(dataclasses.asdict(summary), f)
        mlflow.log_artifact(
            str(summary_path), artifact_path="evaluation/metrics_summary"
        )
        self._logger.info(f"Metrics summary saved to {summary_path}")

    def cleanup_individual_metrics(self) -> None:
        """
        Clears individual metrics from memory to save RAM after CSV export.
        """
        if hasattr(self, "_individual_metrics"):
            delattr(self, "_individual_metrics")
            self._logger.info("Individual metrics cleaned up from memory")

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
                model=self._model.get_pytorch_model(),
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
    ) -> Iterator[XAIExplanationResult]:
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

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            xai_batch: XAIInputBatch = batch

            try:
                results = self.explain_batch(xai_batch, explainer)
            except XAIExplanationError as e:
                failed_batches.append(batch_idx)
                image_names = (
                    xai_batch.image_names
                    if xai_batch.image_names
                    else ["<unknown>"] * len(xai_batch.images_tensor)
                )
                self._logger.warning(f"[Batch {batch_idx}] could not be explained: {e}")
                self._logger.debug(f"[Batch {batch_idx}] image names: {image_names}")
                continue
            except TypeError as e:
                failed_batches.append(batch_idx)
                self._logger.error(f"[Batch {batch_idx}] TypeError: {e}")
                self._logger.debug(
                    "Batch content: " f"{[path for path in xai_batch.image_paths]}"
                )
                continue
            except Exception as e:
                failed_batches.append(batch_idx)
                self._logger.error(f"[Batch {batch_idx}] Unexpected error: {e}")
                self._logger.debug(
                    "Batch content: " f"{[path for path in xai_batch.image_paths]}"
                )
                continue

            for res in results:
                yield res

            if (batch_idx + 1) % 10 == 0:
                self._logger.info(f"Progress: {batch_idx + 1}/{total_batches} batches")

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
            raise XAIExplanationError(f"Explaining batch failed: {e}") from e

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

            result = XAIExplanationResult(
                image=images[i].detach().cpu(),
                image_path=batch.image_paths[i],
                image_name=image_name,
                predicted_class=int(predicted_class),
                prediction_confidence=float(confidence[i].item()),
                true_label=true_label,
                prediction_correct=(predicted_class == true_label),
                topk_predictions=top_k_predictions,
                topk_confidences=top_k_confidences,
                attribution=attributions[i].detach().cpu(),
                explainer_result=explanation_result,
                explainer_name=explainer.__class__.__name__,
                has_bbox=batch.boxes_list[i].numel() > 0,
                bbox=batch.boxes_list[i].detach().cpu(),
                model_name=self._config.model.name,
                processing_time=processing_time / images.size(0),
                timestamp=f"{start_time}",
            )
            results.append(result)

        return results

    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different model.

        Args:
            model_name (str): Name of the new model.
        """
        self._model = self._model_factory.create(model_name)
        self._logger.info(f"Switched to model: {model_name}")

    def get_available_explainers(self) -> List[str]:
        """
        Get the list of available explainers.

        Returns:
            List[str]: Names of available explainers.
        """
        return self._xai_factory.list_available_explainers()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        return self._model.get_model_info()

    def get_run_id(self) -> Optional[str]:
        """
        Get the MLflow run ID if a run is active.

        Returns:
            Optional[str]: MLflow run ID or None if no active run.
        """
        return self._mlflow_run.info.run_id if self._mlflow_run else None

    def quick_test(self, n_samples: int = 5) -> None:
        """
        Quick test with fewer samples by limiting the number of batches processed.

        Args:
            n_samples (int): Number of test samples to process.
        """
        original_max_batches = self._config.data.max_batches
        original_viz = self._config.visualization.save

        batch_size = self._config.data.batch_size
        max_batches_for_samples = math.ceil(n_samples / batch_size)

        self._logger.info(
            f"Quick test: n_samples={n_samples}, batch_size={batch_size}, "
            f"max_batches set to {max_batches_for_samples}"
        )

        self._config.data.max_batches = max_batches_for_samples
        # Disable visualizations during quick test
        self._config.visualization.save = False

        try:
            self.run()
        finally:
            self._config.data.max_batches = original_max_batches
            self._config.visualization.save = original_viz
            self._logger.info("Quick test config restored to original values.")
