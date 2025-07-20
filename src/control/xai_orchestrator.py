import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pipeline_moduls.evaluation.metrics  # noqa: F401
from control.utils.config_dataclasses.master_config import MasterConfig
from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from control.utils.with_cuda_cleanup import with_cuda_cleanup
from pipeline_moduls.data.dataclass.xai_input_batch import XAIInputBatch
from pipeline_moduls.data.image_net_val_dataset import create_dataloader
from pipeline_moduls.data.utils.collate_fn import explain_collate_fn
from pipeline_moduls.evaluation.dataclass.evaluation_summary import EvaluationSummary
from pipeline_moduls.evaluation.xai_evaluator import XAIEvaluator
from pipeline_moduls.models.base.interface.xai_model import XAIModel
from pipeline_moduls.models.base.xai_model_factory import XAIModelFactory
from pipeline_moduls.ResultManager.result_manager import ResultManager
from pipeline_moduls.visualization.visualisation import Visualiser
from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from pipeline_moduls.xai_methods.xai_factory import XAIFactory


class XAIExplanationError(Exception):
    """Custom exception for errors during the XAI explanation process."""

    pass


class XAIOrchestrator:
    """
    Central orchestrator unit of the entire XAI pipeline.
    Coordinates models, datasets, explainers, evaluation, and visualization,
    but does not perform evaluation itself.
    """

    def __init__(self, config: MasterConfig):
        """
        Initialize the XAI Orchestrator with the given configuration.

        Args:
            config: MasterConfig object loaded via Hydra.
        """
        self._individual_metrics = None
        self._mlflow_run = None
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._result_manager = ResultManager()

        # Setup factories
        self._model_factory: XAIModelFactory = XAIModelFactory()
        self._xai_factory: XAIFactory = XAIFactory()

        # Load model
        self._model_name: str = self._config.model.name
        self._model: XAIModel = self._model_factory.create(self._model_name)
        self._pytorch_model = self._model.get_pytorch_model()
        self._device = next(self._pytorch_model.parameters()).device

        # Evaluator and Visualizer
        self._evaluator = XAIEvaluator()
        self._visualiser = Visualiser(
            show=self._config.visualization.show,
            save_path=Path(self._config.experiment.output_dir),
        )

        self._logger.info("Orchestrator initialized:")
        self._logger.info(f"  Model: {self._model_name} on {self._device}")
        self._logger.info(
            f"  Available explainers: {self._xai_factory.list_available_explainers()}"
        )

    def run(self):
        self.prepare_experiment()
        dataloader = self.setup_dataloader()
        explainer = self.create_explainer(
            explainer_name=self._config.xai.name, **self._config.xai.kwargs
        )
        results = self.run_pipeline(dataloader, explainer)
        summary = self.evaluate_results(results)
        self.save_results(results, summary)
        self.visualize_results_if_needed(results, summary)
        self.cleanup_individual_metrics()
        self.finalize_run()

    def prepare_experiment(self):
        """
        Starts the MLflow experiment and logs parameters.
        """
        self._logger.info(f"Starting experiment: {self._config.experiment.name}")
        self._mlflow_run = mlflow.start_run(run_name=self._config.experiment.name)

        mlflow.log_param("model_name", self._model_name)
        mlflow.log_param("explainer_name", self._config.xai.name)
        mlflow.log_param("batch_size", self._config.data.batch_size)
        mlflow.log_param("max_batches", self._config.data.max_batches)

    def run_pipeline(
        self,
        dataloader: DataLoader,
        explainer: BaseExplainer,
    ) -> List[XAIExplanationResult]:
        """runs the pipeline: process data loader batches through the explainer.

        Args:
            dataloader (DataLoader): DataLoader providing batches to explain.
            explainer (BaseExplainer): The explainer instance to generate explanations.

        Returns:
            List[XAIExplanationResult]: List of explanation results for all images.
        """
        self._logger.info(f"Processing samples with {self._config.xai.name}...")
        results = []

        for result in self.process_dataloader(
            dataloader=dataloader,
            explainer=explainer,
            max_batches=self._config.data.max_batches,
        ):
            results.append(result)
            self._result_manager.add_results("step_1", [result])

        self._logger.info(
            f"Finished processing. Total results collected: {len(results)}"
        )
        return results

    def evaluate_results(
        self, results: List[XAIExplanationResult]
    ) -> EvaluationSummary:
        """
        Evaluates a batch of explanation results and logs metrics to MLflow.

        Args:
            results (List[XAIExplanationResult]): List of explanation results.

        Returns:
            EvaluationSummary: Aggregated evaluation summary.
        """
        self._logger.info("Calculating evaluation metrics...")

        # Sammle individuelle Metriken während der Evaluation
        individual_metrics = []
        correct_predictions = 0
        total_processing_time = 0

        self._logger.info(
            f"Processing {len(results)} results for individual metrics..."
        )

        for i, result in enumerate(results):
            # Prediction Accuracy
            if result.prediction_correct is not None and result.prediction_correct:
                correct_predictions += 1

            total_processing_time += result.processing_time

            # Berechne individuelle XAI Metriken
            metrics = self._evaluator.evaluate_single_result(result)
            individual_metrics.append(metrics)

            # Progress Logging
            if (i + 1) % 10 == 0:
                self._logger.info(
                    f"Processed {i + 1}/{len(results)} individual metrics"
                )

        # Speichere individuelle Metriken für später
        self._individual_metrics = individual_metrics

        # Erstelle Summary aus bereits berechneten individuellen Metriken
        summary = self._evaluator.create_summary_from_individual_metrics(
            results=results,
            individual_metrics=individual_metrics,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

        self._logger.info("Evaluation metrics calculation finished!")

        # Log metrics to MLflow
        for key, value in summary.to_dict().items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        return summary

    def save_results(
            self, results: List[XAIExplanationResult], summary: EvaluationSummary
    ):
        """saves results and evaluation summary to disk and logs artifacts to MLflow.

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

        # Summary speichern
        summary_path = output_dir / "metrics_summary.yaml"
        with open(summary_path, "w") as f:
            import yaml
            yaml.dump(dataclasses.asdict(summary), f)
        mlflow.log_artifact(
            str(summary_path), artifact_path="evaluation/metrics_summary"
        )
        self._logger.info(f"Metrics summary saved to {summary_path}")

    def cleanup_individual_metrics(self):
        """Bereinige individuelle Metriken nach CSV-Export um memory zu sparen"""
        if hasattr(self, "_individual_metrics"):
            delattr(self, "_individual_metrics")
            self._logger.info("Individual metrics cleaned up from memory")

    def visualize_results_if_needed(
        self, results: List[XAIExplanationResult], summary: EvaluationSummary
    ):
        """
        Generates and logs visualizations if configured to do so.

        Args:
            results (List[XAIExplanationResult]): List of explanation results.
            summary (EvaluationSummary): Evaluation summary.
        """
        if not self._config.visualization.save:
            return

        self._logger.info("Generating visualizations...")
        for result in results:
            if result.attribution is None:
                if result.attribution_path and Path(result.attribution_path).exists():
                    result.attribution = torch.load(result.attribution_path)
                else:
                    self._logger.warning(
                        f"Attribution for {result.image_name} could not be loaded "
                        f"(path: {result.attribution_path})"
                    )

            metrics = summary
            vis_path = self._visualiser.create_visualization(
                result=result, metrics=metrics
            )

            if vis_path:
                mlflow.log_artifact(
                    vis_path,
                    artifact_path=f"visualizations/{self._model_name}/"
                    f"{self._config.xai.name}",
                )

    def finalize_run(self):
        """
        Ends the MLflow run if active and logs the run ID.
        """
        if self._mlflow_run:
            mlflow.end_run()
            self._logger.info(f"MLflow run ended: {self._mlflow_run.info.run_id}")
            self._mlflow_run = None

    def setup_dataloader(
        self,
        project_root: Optional[Path] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = False,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        transform: Optional[transforms.Compose] = None,
    ) -> DataLoader:
        """
        Sets up the ImageNet DataLoader with configurable parameters.

        Args:
            project_root (Optional[Path]): Root path of the project.
                If None, it is inferred automatically.
            batch_size (int): Batch size for loading the dataset.
            num_workers (int): Number of subprocesses used for data loading.
            pin_memory (bool): Whether to pin memory (recommended for GPU).
            shuffle (bool): Whether to shuffle the dataset during loading.
            target_size (Optional[Tuple[int, int]]): Target size for resizing images
                (e.g., (224, 224)).
            transform (Optional[transforms.Compose]): Optional custom
                transform to apply to images.

        Returns:
            DataLoader: Configured PyTorch DataLoader with ImageNet validation data.

        Notes:
            Uses `create_dataloader()` internally and expects it to be compatible with
            `ImageNetSample` output and the `explain_collate_fn`.
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
            f"DataLoader setup complete: {len(dataloader.dataset)} samples in "
            f"{len(dataloader)} batches "
            f"(batch_size={batch_size}, shuffle={shuffle})"
        )

        return dataloader

    def create_explainer(
        self, explainer_name: str, **additional_kwargs
    ) -> BaseExplainer:
        """
        Creates an explainer with runtime parameter validation.

        Args:
            explainer_name: Name of the explainer.
            **kwargs: Additional parameters.

        Returns:
            Configured explainer instance.
        """
        logger = logging.getLogger(__name__)

        # 1. Kombiniere Config-Parameter mit zusätzlichen kwargs
        config_kwargs = self._config.xai.kwargs.copy()
        config_kwargs.update(additional_kwargs)

        # 2. Füge use_defaults Flag hinzu
        use_defaults = self._config.xai.use_defaults

        # 3. Log was übergeben wird (für Debugging)
        logger.info(f"Creating {explainer_name} explainer...")
        logger.info(f"Use defaults: {use_defaults}")
        logger.debug(f"Parameters: {config_kwargs}")

        # 4. Erstelle Explainer (Validierung passiert automatisch im __init__)
        try:
            explainer = self._xai_factory.create_explainer(
                name=explainer_name,
                model=self._model.get_pytorch_model(),
                use_defaults=use_defaults,
                **config_kwargs,
            )

            # 5. Log Validierung-Ergebnis (wurde bereits im Explainer geloggt)
            validation_result = explainer.config_validation

            if validation_result.status == ValidationResult.MISSING_USING_DEFAULTS:
                logger.info("Some parameters used default values. See warnings above.")

            elif validation_result.status == ValidationResult.VALID:
                logger.info(
                    "Explainer created successfully with all parameters validated."
                )

            return explainer

        except ValueError as e:
            # Config validation failed
            logger.error(f"Failed to create {explainer_name}: {e}")
            logger.error("Check your config parameters or set 'use_defaults: true'")
            raise
        except Exception as e:
            # Other errors
            logger.error(f"Unexpected error creating {explainer_name}: {e}")
            raise

    @with_cuda_cleanup
    def process_dataloader(
        self,
        dataloader: DataLoader,
        explainer: BaseExplainer,
        max_batches: Optional[int] = None,
    ) -> Iterator[XAIExplanationResult]:
        """
        Iterates over a DataLoader and explains each batch using a given XAI explainer.
        Yields XAIExplanationResult objects for each image in a memory-efficient way.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader that yields batches created by
                a custom `collate_fn`. The collate function must return a tuple of:
                (images_tensor, labels_tensor, boxes_list, image_paths, image_names,
                bbox_paths, labels_int).
            explainer (BaseExplainer): The configured XAI explainer used to generate
                attributions.
            max_batches (Optional[int]): Optional limit on the number of batches to
                process.

        Yields:
            XAIExplanationResult: One result per image in the batch.
        """
        total_batches = min(len(dataloader),max_batches)
        self._logger.info(f"Starting processing of {total_batches} batches...")

        for batch_idx, batch in enumerate(dataloader):
            batch:XAIInputBatch = batch
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:
                results = self.explain_batch(batch,explainer)
                for res in results:
                    yield res

                if (batch_idx + 1) % 10 == 0:
                    self._logger.info(
                        f"Progress: {batch_idx + 1}/{total_batches} batches"
                    )
            except TypeError as e:
                self._logger.error(f"[Batch {batch_idx}] TypeError: {e}")
                self._logger.debug(
                    f"Batch content: {[path for path in batch.image_paths]}")  # image_paths
            except Exception as e:
                self._logger.error(f"[Batch {batch_idx}] Unexpected error: {e}")
                self._logger.debug(
                    f"Batch content: {[path for path in batch.image_paths]}")  # image_paths

    @with_cuda_cleanup
    def explain_batch(
            self,
            batch: XAIInputBatch,
            explainer: BaseExplainer
    ) -> List[XAIExplanationResult]:
        """
        Explains a batch of images using the provided XAI explainer.

        Args:
            batch: Expected to be a tuple:
                (images_tensor, labels_tensor, boxes_list, image_paths, image_names,
                bbox_paths, labels_int)
            explainer: Configured explainer instance.

        Returns:
            List of XAIExplanationResult instances.

        Raises:
            TypeError: If batch does not match the expected structure.
        """
        start_time = time.time()

        images = batch.images_tensor.to(self._device)
        labels_tensor = batch.labels_tensor.to(self._device)

        explanation_result = None
        try:
            explanation_result = explainer.explain(images,
                                                   labels_tensor)
            attributions = explanation_result.attributions  # [B, C, H, W]
            pred_classes = (
                explanation_result.predictions
            )  # e.g. [B] or [B, num_classes]
            target_labels = explanation_result.target_labels  # [B]

        except Exception as e:
            self._logger.error(f"Error explaining batch: {e}")
            self._logger.warning("Using dummy explanation result due to error.")
            b = images.size(0)
            attributions = torch.zeros_like(images)
            pred_classes = torch.full((b,), -1)
            target_labels = labels_tensor
            explanation_result = None #
        processing_time = time.time() - start_time

        results = []
        for i in range(images.size(0)):
            if pred_classes.dim() == 2:
                predicted_class = pred_classes[i].argmax().item()
            else:
                predicted_class = pred_classes[i].item()

            true_label = int(target_labels[i])
            image_name = batch.image_names[i] if batch.image_names else f"image_{i}"

            result = XAIExplanationResult(
                image=images[i].detach().cpu(),
                image_path=batch.image_paths[i],
                image_name=image_name,
                predicted_class=predicted_class,
                true_label=true_label,
                prediction_correct=(predicted_class == true_label),
                attribution=attributions[i].detach().cpu(),
                explainer_result=explanation_result,
                explainer_name=explainer.__class__.__name__,
                has_bbox=batch.boxes_list[i].numel() > 0,
                bbox=batch.boxes_list[i].detach().cpu(),
                bbox_info=batch.boxes_list[i].detach().cpu(),
                model_name=self._model_name,
                processing_time=processing_time / images.size(0),
                timestamp=f"{start_time}",
                error = str(e) if explanation_result is None else None
            )
            results.append(result)

        return results

    def switch_model(self, model_name: str):
        """
        Switch to a different model.

        Args:
            model_name: Name of the new model.
        """
        self._model = self._model_factory.create(model_name)
        self._pytorch_model = self._model.get_pytorch_model()
        self._device = next(self._pytorch_model.parameters()).device
        self._model_name = model_name

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
        self._config.visualization.save = False  # Disable visualizations during
        # quick test

        try:
            self.run()
        finally:
            self._config.data.max_batches = original_max_batches
            self._config.visualization.save = original_viz
            self._logger.info("Quick test config restored to original values.")
