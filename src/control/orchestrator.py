import dataclasses
import logging
import time
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
from src.pipeline_moduls.evaluation.streaming_evaluator import StreamingEvaluator
from src.pipeline_moduls.data.image_net_label_mapper import ImageNetLabelMapper
from src.pipeline_moduls.metaanlyse.xai_meta_analysis import XaiMetaAnalysis
from src.pipeline_moduls.models.base.xai_model import XAIModel
from src.pipeline_moduls.resultmanager.result_manager import ResultManager
from src.control.utils.config_dataclasses.master_config import MasterConfig
from src.control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from src.control.utils.error.xai_explanation_error import XAIExplanationError
from src.control.utils.set_up_logger import setup_logger
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
from src.pipeline_moduls.models.xai_model_factory import XAIModelFactory
from src.pipeline_moduls.visualization.visualisation import Visualiser
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.xai_factory import XAIFactory
from utils.with_cuda_cleanup import with_cuda_cleanup


class Orchestrator:
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
        self._individual_metrics: List[Any] = []
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
        self._streaming_evaluator = StreamingEvaluator(
            base_evaluator=self._evaluator,
            logger=self._logger
        )
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

    def process_dataset_streaming(
            self,
            dataset,
            explainer: BaseExplainer,
            max_samples: Optional[int] = None,
            batch_size: int = 50
    ) -> Iterator[XAIExplanationResult]:
        """
        Process dataset and yield XAI results one by one for streaming evaluation.

        Args:
            dataset: Dataset to process
            explainer: XAI explainer to use
            max_samples: Maximum number of samples to process
            batch_size: Batch size for model inference (not evaluation batching)

        Yields:
            XAIExplanationResult: Individual processed results
        """
        self._logger.info(f"Starting streaming dataset processing...")

        dataloader = self.setup_dataloader(
            project_root=None,
            batch_size=self._config.data.batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
            shuffle=self._config.data.shuffle,
            target_size=self._config.data.resize,
            transform=None,
        )

        total_processed = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if max_samples and total_processed >= max_samples:
                break

            # Process batch and yield individual results
            for result in self._process_batch_streaming(batch, explainer):
                total_processed += 1
                yield result

                if max_samples and total_processed >= max_samples:
                    break

        self._logger.info(f"Streaming processing completed: {total_processed} results")

    def _process_batch_streaming(
            self,
            batch: XAIInputBatch,
            explainer: BaseExplainer
    ) -> Iterator[XAIExplanationResult]:
        """
        Process a single batch and yield individual results.

        Args:
            batch: Input batch
            explainer: XAI explainer

        Yields:
            XAIExplanationResult: Individual processed results
        """
        try:
            # Model predictions for the batch
            with torch.no_grad():
                predictions = self._model.pytorch_model(batch.images_tensor.to(
                    self._config.hardware.device))
                probabilities = torch.softmax(predictions, dim=1)
                predicted_classes = torch.argmax(predictions, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

            # Process each item in the batch
            for i in range(batch.images_tensor.size(0)):
                try:
                    result = self._process_single_item_streaming(
                        batch, i, explainer, predicted_classes[i], confidences[i]
                    )
                    if result:
                        yield result

                except Exception as e:
                    self._logger.error(f"Failed to process batch item {i}: {e}")
                    continue

        except Exception as e:
            self._logger.error(f"Failed to process batch: {e}")
            return

    def _process_single_item_streaming(
            self,
            batch: XAIInputBatch,
            item_idx: int,
            explainer: BaseExplainer,
            predicted_class: torch.Tensor,
            confidence: torch.Tensor
    ) -> Optional[XAIExplanationResult]:
        """
        Process a single item from batch and return XAI result.
        Memory-optimized version that doesn't store large tensors.
        """
        start_time = time.time()

        try:
            # Extract single item data from XAIInputBatch
            single_image = batch.images_tensor[
                           item_idx:item_idx + 1]  # Keep batch dimension
            true_label = batch.labels_int[item_idx]  # Use labels_int list, not tensor
            image_name = batch.image_names[item_idx]
            image_path = batch.image_paths[item_idx]

            # Generate explanation
            explainer_result = explainer.explain(
                images=single_image,
                target_labels=predicted_class.unsqueeze(0),
                top_k=10
            )

            # Extract attribution tensor (this is the memory-heavy part)
            attribution = explainer_result.attributions[0].cpu()

            # Get bbox info from XAIInputBatch structure
            bbox_info = {}
            has_bbox = False

            # Check if we have bounding box data for this item
            if (item_idx < len(batch.boxes_list) and
                    batch.boxes_list[item_idx] is not None and
                    len(batch.boxes_list[item_idx]) > 0):
                bbox_tensor = batch.boxes_list[item_idx]  # Tensor with shape [N, 4]
                bbox_path = batch.bbox_paths[item_idx]

                bbox_info = {
                    'bbox_tensor': bbox_tensor,  # Raw bounding box coordinates
                    'bbox_path': bbox_path,  # Path to XML annotation file
                    'num_objects': len(bbox_tensor)  # Number of objects in image
                }
                has_bbox = True

            # Create result
            processing_time = time.time() - start_time

            result = XAIExplanationResult(
                image=None,  # Don't store image tensor to save memory
                image_name=image_name,
                image_path=image_path,
                predicted_class=predicted_class.item(),
                true_label=true_label,
                prediction_confidence=confidence.item(),
                prediction_correct=(predicted_class.item() == true_label),
                attribution=attribution,
                explainer_result=explainer_result,
                explainer_name=explainer.get_name(),
                has_bbox=has_bbox,
                bbox_info=bbox_info,
                bbox=batch.boxes_list[item_idx] if has_bbox else None,
                model_name=self._config.model.name,
                processing_time=processing_time
            )

            return result

        except Exception as e:
            self._logger.error(f"Error processing item {item_idx}: {e}")
            return None

    def run_streaming_evaluation(
            self,
            dataset,
            explainer: BaseExplainer,
            max_samples: Optional[int] = None,
            evaluation_batch_size: int = 50,
            store_individual_for_csv: bool = True
    ) -> EvaluationSummary:
        """
        Run complete pipeline with streaming evaluation.

        Args:
            dataset: Dataset to process
            explainer: XAI explainer
            max_samples: Maximum samples to process
            evaluation_batch_size: Batch size for evaluation progress updates
            store_individual_for_csv: Whether to store individual metrics for CSV

        Returns:
            EvaluationSummary: Final evaluation summary
        """
        self._logger.info("Starting streaming evaluation pipeline...")

        # Reset streaming evaluator
        self._streaming_evaluator.reset_for_new_run(
            store_individual=store_individual_for_csv
        )

        # Create result stream
        result_stream = self.process_dataset_streaming(
            dataset=dataset,
            explainer=explainer,
            max_samples=max_samples,
            batch_size=32  # Model inference batch size
        )

        # Process stream with progress updates
        progress_updates = self._streaming_evaluator.process_result_stream(
            result_stream=result_stream,
            batch_size=evaluation_batch_size
        )

        # Log progress updates
        for progress in progress_updates:
            self._log_progress_update(progress)

            # Log intermediate metrics to MLflow
            if progress["current_metric_averages"]:
                for metric_name, value in progress["current_metric_averages"].items():
                    mlflow.log_metric(f"running_{metric_name}", value,
                                      step=progress["total_processed"])

        # Create final summary
        summary = self._streaming_evaluator.create_final_summary()

        # Log final metrics to MLflow
        for key, value in summary.to_dict().items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final_{key}", value)

        self._logger.info("Streaming evaluation completed!")
        return summary

    def _log_progress_update(self, progress: Dict[str, Any]) -> None:
        """Log progress update information"""
        self._logger.info(
            f"Batch {progress['batch_number']}: "
            f"Processed {progress['total_processed']} samples "
            f"({progress['samples_with_bbox']} with bbox), "
            f"Accuracy: {progress['prediction_accuracy']:.3f}, "
            f"Avg time: {progress['avg_processing_time']:.3f}s"
        )

        if progress["current_metric_averages"]:
            self._logger.info("Current metric averages:")
            for metric, value in progress["current_metric_averages"].items():
                self._logger.info(f"  {metric}: {value:.4f}")

    def save_streaming_results(self, summary: EvaluationSummary) -> None:
        """
        Save streaming evaluation results.
        Only saves individual metrics if they were stored.
        """
        output_dir = Path(self._config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = self._result_manager.save_evaluation_summary_to_file(
            summary, output_dir
        )
        mlflow.log_artifact(str(summary_path),
                            artifact_path="evaluation/metrics_summary")

        # Save CSV only if individual metrics were stored
        individual_metrics = self._streaming_evaluator.get_individual_metrics_for_csv()
        if individual_metrics:
            self._logger.info("Saving individual metrics to CSV...")
            csv_path = self._result_manager.save_dataframe_with_metrics(
                path=output_dir,
                individual_metrics=individual_metrics,
            )
            mlflow.log_artifact(str(csv_path), artifact_path="evaluation/csv_results")
        else:
            self._logger.info(
                "Individual metrics were not stored - skipping CSV export")


    def run(self) -> Dict[str, Any]:
        """
        Main run method using streaming evaluation
        """
        try:
            self._pipeline_status = "running"
            self._current_step = "setup"

            # Setup
            dataset = self.setup_dataloader(
                project_root=None,
                batch_size=self._config.data.batch_size,
                num_workers=self._config.data.num_workers,
                pin_memory=self._config.data.pin_memory,
                shuffle=self._config.data.shuffle,
                target_size=self._config.data.resize,
                transform=None,
            )
            explainer = self.create_explainer(self._config.xai.name,
                                              self._config.xai.kwargs,
                                              self._config.xai.use_defaults)

            # Run streaming evaluation
            self._current_step = "evaluation"
            summary = self.run_streaming_evaluation(
                dataset=dataset,
                explainer=explainer,
                max_samples=self._config.data.batch_size*self._config.data.max_batches,
                evaluation_batch_size=50,  # Evaluation progress batch size
                store_individual_for_csv=True
            )

            # Save results
            self._current_step = "saving"
            self.save_streaming_results(summary)

            self._pipeline_status = "completed"
            self._current_step = "finished"

            return {
                "status": "success",
                "summary": summary.to_dict(),
                "total_samples": summary.total_samples,
                "samples_with_bbox": summary.samples_with_bbox
            }

        except Exception as e:
            self._pipeline_error = e
            self._pipeline_status = "failed"
            self._logger.error(f"Pipeline failed: {e}")
            raise

    def run_explanation_pipeline(
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

        max_batches = self._config.data.max_batches
        for batch in self.process_dataloader(
            dataloader=dataloader,
            explainer=explainer,
            max_batches=max_batches,
        ):
            if self._config.model.transform:
                transformed_batch = [self.transform_result(r) for r in batch]
            else:
                transformed_batch = batch

            results.extend(transformed_batch)

        self._result_manager.add_results(results)
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

        for i, result in enumerate(tqdm(results, desc="Processing results")):
            if result.prediction_correct:
                correct_predictions += 1

            total_processing_time += result.processing_time

            metrics = self._evaluator.evaluate_single_result(result)
            individual_metrics.append(metrics)

        self._individual_metrics.extend(individual_metrics)

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

    def save_results(self, summary: EvaluationSummary) -> None:
        """
        Saves results and evaluation summary to disk and logs artifacts to MLflow.

        Args:
            summary (EvaluationSummary): Evaluation summary object.
        """
        output_dir = Path(self._config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        individual_metrics = self._individual_metrics
        self._logger.info(f"individual_metics: {individual_metrics}")

        csv_path = self._result_manager.save_dataframe_with_metrics(
            path=output_dir,
            individual_metrics=self._individual_metrics,
        )

        mlflow.log_artifact(str(csv_path), artifact_path="evaluation/csv_results")
        self._logger.info(f"CSV with metrics saved to {csv_path}")

        # Save summary
        summary_path = self._result_manager.save_evaluation_summary_to_file(
            summary, output_dir
        )
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

    def finalize_run(self) -> None:
        """
        Ends the MLflow run if active and logs the run ID.
        """
        if self._mlflow_run and mlflow.active_run() is not None:
            mlflow.end_run()
            self._logger.info(f"MLflow run ended: {self._mlflow_run.info.run_id}")
            self._mlflow_run = None

        self.cleanup_individual_metrics()

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
                else:
                    yield results
                pbar.update(1)

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

    def transform_result(self, result: XAIExplanationResult) -> XAIExplanationResult:
        """
        Applies class label transformation and enriches the result with metadata.

        This method maps the predicted class to a validation class index,
        sets readable class names for both predicted and true labels,
        determines if the prediction was correct, and stores the original
        predicted class before transformation.

        Note:
            This method assumes transformation is required.
            The caller must ensure it is only called if `transform=True`.

        Args:
            result (XAIExplanationResult): The explanation result to transform.

        Returns:
            XAIExplanationResult: A copy of the result with transformed labels,
            class names, and a correctness flag.
        """
        label_lookup = self.label_mapper.class_id_to_label
        class_to_val_tensor = self.label_mapper.class_to_val_tensor
        orig_class: int = result.predicted_class
        mapped_class: int = int(class_to_val_tensor[orig_class])

        true_label_val_idx = result.true_label
        true_label_name = label_lookup.get(
            true_label_val_idx, f"Class {true_label_val_idx}"
        )

        prediction_correct = (
            (mapped_class == true_label_val_idx)
            if true_label_val_idx is not None
            else False
        )

        return dataclasses.replace(
            result,
            predicted_class=mapped_class,
            predicted_class_before_transform=orig_class,
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
        grouped, threshold_fig = analysis.threshold_analysis("iou")
        threshold_img_path = plot_dir / "threshold_iou_score.png"
        threshold_fig.savefig(threshold_img_path)
        plt.close(threshold_fig)
        mlflow.log_artifact(
            str(threshold_img_path), artifact_path="meta_analysis/plots"
        )

        # Gruppierte Daten als CSV speichern & loggen
        threshold_csv_path = meta_dir / "threshold_iou_score.csv"
        grouped.to_csv(threshold_csv_path)
        mlflow.log_artifact(str(threshold_csv_path), artifact_path="meta_analysis")

        self._logger.info(f"Meta-Analyse gespeichert in {meta_dir.resolve()}")

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

    def prepare_experiment(self) -> None:
        """
        Starts the MLflow experiment and logs parameters.
        """
        self._logger.info(f"Starting experiment: {self._config.experiment.name}")
        if mlflow.active_run() is None:
            self._mlflow_run = mlflow.start_run(run_name=self._config.experiment.name)
        else:
            self._mlflow_run = mlflow.active_run()

        mlflow.pytorch.log_model(
            self._model.pytorch_model,
            name="model",
            registered_model_name=self._config.model.name,
        )
        mlflow.log_param("explainer_name", self._config.xai.name)
        mlflow.log_param("batch_size", self._config.data.batch_size)
        mlflow.log_param("max_batches", self._config.data.max_batches)

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
        self, explainer_name: str, explainer_parameters: Any, use_defaults: bool
    ) -> BaseExplainer:
        """
        Creates an explainer with runtime parameter validation.

        Args:
            explainer_name (str): Name of the explainer (e.g., 'grad_cam').
            explainer_parameters (dict): Parameters for the chosen explainer.
                Passed as keyword arguments to the explainer's constructor.
            use_defaults (bool): If True, unspecified parameters will be filled with
                default values.
        Returns:
            BaseExplainer: Configured explainer instance.
        Raises:
            TypeError, ValueError, Exception: If creation fails.
        """
        logger = logging.getLogger(__name__)

        config_kwargs = explainer_parameters

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
