import dataclasses
import logging
import time
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import mlflow
import pandas as pd
from mlflow import MlflowClient
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
from tqdm import tqdm

import src.pipe.moduls.evaluation.metrics  # noqa: F401
from src.pipe.control.utils.config_dataclasses.master_config import MasterConfig
from src.pipe.control.utils.dataclasses.xai_explanation_result import (
    XAIExplanationResult,
)
from src.pipe.control.utils.error.xai_explanation_error import XAIExplanationError
from src.pipe.control.utils.set_up_logger import setup_logger
from src.pipe.moduls.data.dataclass.xai_input_batch import XAIInputBatch
from src.pipe.moduls.data.image_net_label_mapper import ImageNetLabelMapper
from src.pipe.moduls.data.image_net_val_dataset import (
    ImageNetValDataset,
    create_dataloader,
)
from src.pipe.moduls.data.utils.collate_fn import explain_collate_fn
from src.pipe.moduls.evaluation.dataclass.evaluation_summary import (
    EvaluationSummary,
)
from src.pipe.moduls.evaluation.xai_evaluator import XAIEvaluator
from src.pipe.moduls.models.base.xai_model import XAIModel
from src.pipe.moduls.models.xai_model_factory import XAIModelFactory
from src.pipe.moduls.resultmanager.result_manager import ResultManager
from src.pipe.moduls.single_run_analyse.single_run_analysis import (
    SingleRunAnalyse,
)
from src.pipe.moduls.visualization.visualisation import Visualiser
from src.pipe.moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipe.moduls.xai_methods.xai_factory import XAIFactory
from src.pipe.utils.with_cuda_cleanup import with_cuda_cleanup


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

        project_root = Path(__file__).resolve().parents[3]
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

        # Load model with seed from experiment config
        self._model: XAIModel = self._model_factory.create(
            config.model.name, seed=config.experiment.seed
        )

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
                explainer_name=self._config.xai.name,
                explainer_parameters=self._config.xai.kwargs,
                use_defaults=self._config.xai.use_defaults,
            )

            self._current_step = "pipeline_execution"
            self._logger.info(f"Starting step: {self._current_step}")
            xai_results = self.run_explanation_pipeline(dataloader, explainer)
            self._logger.debug(
                "XAI result example after transformation:" f" {xai_results[0]}"
            )
            # in einem Batch process durchlaufen wird
            self._current_step = "results_evaluation"
            self._logger.info(f"Starting step: {self._current_step}")
            summary = self.evaluate_results(xai_results)

            self._current_step = "results_saving"
            self._logger.info(f"Starting step: {self._current_step}")
            self.save_results(summary)

            self._current_step = "single run analysis"
            self._logger.info(f"Starting step: {self._current_step}")
            self.xai_single_run_analyse()

            self._current_step = "visualization"
            self._logger.info(f"Starting step: {self._current_step}")
            self.visualize_results_if_needed(xai_results, summary)
        except Exception as e:
            self._pipeline_status = "failed"
            self._pipeline_error = e
            self._logger.error(f"PIPELINE FAILED at step: {self._current_step}")
            self._logger.error(f"Error: {str(e)}")
            raise
        else:
            self._logger.info("Pipeline completed successfully!")
            self._pipeline_status = "completed"
            self._current_step = "completed"
        finally:
            self._current_step = "finalization"
            self._logger.info(f"Starting step: {self._current_step}")
            self.finalize_run()

        return {
            "status": self._pipeline_status,
            "total_samples": len(xai_results),
            "output_dir": self._config.experiment.output_dir,
            "explainer": self._config.xai.name,
            "_model": self._config.model.name,
        }

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
        Uses optimized batch processing for better performance.
        """
        self._logger.info("Calculating evaluation metrics using batch processing...")

        # Calculate prediction accuracy and processing time
        correct_predictions = 0
        total_processing_time = 0.0

        for result in results:
            if result.prediction_correct:
                correct_predictions += 1
            total_processing_time += result.processing_time

        self._logger.info(f"Using batch evaluation for {len(results)} results...")

        # Use optimized batch processing instead of individual processing
        individual_metrics = self._evaluator.evaluate_batch_metrics(results)

        # Store metrics for later use
        self._individual_metrics.extend(individual_metrics)

        summary = self._evaluator.create_summary_from_individual_metrics(
            results=results,
            individual_metrics=individual_metrics,
            correct_predictions=correct_predictions,
            total_processing_time=total_processing_time,
        )

        self._logger.info("Batch evaluation metrics calculation finished!")

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
        self._logger.debug(f"Individual metrics: {len(individual_metrics)} results")

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
            self._logger.debug("Individual metrics cleaned up from memory")

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

    def xai_single_run_analyse(self):
        """
        Run simplified single-run analysis and store histograms and basic CSV data.
        Results are written to:
            - results/<experiment>/single_run_analysis/plots/ (histograms)
            - results/<experiment>/single_run_analysis/data/ (CSV files)
        """
        output_dir = Path(self._config.experiment.output_dir)
        csv_path = output_dir / "results_with_metrics.csv"

        if not csv_path.exists():
            self._logger.error(f"Expected CSV not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)
        analysis = SingleRunAnalyse(df)

        meta_dir = output_dir / "single_run_analysis"
        plot_dir = meta_dir / "plots"
        data_dir = meta_dir / "data"
        plot_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info("Starting single-run analysis...")

        # 1. IoU histograms by correctness
        self._logger.info("Creating IoU histograms...")
        try:
            iou_plots = analysis.plot_iou_histograms_by_correctness(data_dir)
            for plot_name, plot_path in iou_plots.items():
                mlflow.log_artifact(str(plot_path), artifact_path="single_run_analysis/plots")
            self._logger.info(f"Created {len(iou_plots)} IoU histogram plots")
        except Exception as e:
            self._logger.warning(f"IoU histograms failed: {e}")

        # 2. Other metric histograms
        self._logger.info("Creating prediction correctness histograms...")
        try:
            other_plots = analysis.plot_prediction_correctness_histograms(data_dir)
            for plot_name, plot_path in other_plots.items():
                mlflow.log_artifact(str(plot_path), artifact_path="single_run_analysis/plots")
            self._logger.info(
                f"Created {len(other_plots)} prediction correctness histogram plots"
            )
        except Exception as e:
            self._logger.warning(f"Prediction correctness histograms failed: {e}")

        # 3. Pixel Precision histograms
        self._logger.info("Creating pixel precision histograms...")
        try:
            pixel_precision_plots = (
                analysis.plot_pixel_precision_histograms_by_correctness(data_dir)
            )
            for plot_name, plot_path in pixel_precision_plots.items():
                mlflow.log_artifact(str(plot_path),
                                    artifact_path="single_run_analsis/plots")
            self._logger.info(
                f"Created {len(pixel_precision_plots)} pixel precision histogram plots"
            )
        except Exception as e:
            self._logger.warning(f"Pixel precision histograms failed: {e}")

        # 4. Pixel Recall histograms
        self._logger.info("Creating pixel recall histograms...")
        try:
            pixel_recall_plots = analysis.plot_pixel_recall_histograms_by_correctness(
                data_dir
            )
            for plot_name, plot_path in pixel_recall_plots.items():
                mlflow.log_artifact(str(plot_path),
                                    artifact_path="single_run_analysis/plots")
            self._logger.info(
                f"Created {len(pixel_recall_plots)} pixel recall histogram plots"
            )
        except Exception as e:
            self._logger.warning(f"Pixel recall histograms failed: {e}")

        # 5. Basic correlations (for simple_analyzer to use)
        self._logger.info("Computing correlations...")
        corrs = analysis.correlation_with_correctness()
        corr_df = pd.DataFrame(
            list(corrs.items()), columns=["metric", "correlation_with_correctness"]
        )
        corr_csv_path = data_dir / "correlations.csv"
        corr_df.to_csv(corr_csv_path, index=False)
        mlflow.log_artifact(str(corr_csv_path), artifact_path="single_run_analysis/data")

        # Log all CSV data files from histograms
        for data_file in data_dir.glob("*_histogram_data.csv"):
            mlflow.log_artifact(str(data_file), artifact_path="single_run_analysis/data")

        self._logger.info(f"Single-run analysis completed - Results in: {meta_dir}")

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
        # Configure MLflow from config
        mlflow.set_tracking_uri(self._config.mlflow.tracking_uri)
        mlflow.set_experiment(self._config.mlflow.experiment_name)

        # Enable autologging if configured
        if self._config.mlflow.auto_log:
            mlflow.pytorch.autolog()

        self._logger.info(f"Starting experiment: {self._config.experiment.name}")
        if mlflow.active_run() is None:
            # Use configured run_name or generate default
            run_name = (
                self._config.mlflow.run_name
                or f"{self._config.experiment.name}_{self._config.model.name}_"
                f"{self._config.xai.name}"
            )

            # Prepare run parameters
            run_kwargs = {"run_name": run_name}

            # Add tags if configured
            if self._config.mlflow.tags:
                run_kwargs["tags"] = self._config.mlflow.tags

            self._mlflow_run = mlflow.start_run(**run_kwargs)
            self._logger.info(
                f"Started MLflow run: {self._mlflow_run.info.run_id} with name: {run_name}"
            )
        else:
            self._mlflow_run = mlflow.active_run()
            self._logger.info(
                f"Using existing MLflow run: {self._mlflow_run.info.run_id}"
            )

        # Log comprehensive configuration parameters
        self._log_comprehensive_config()

        # Only log model if it's not already registered or if explicitly requested
        self._log_model_conditionally()

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
            project_root = Path(__file__).resolve().parents[3]

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
        logger.info(f"Parameters: {config_kwargs}")
        logger.info(f"Model: {self._model.__class__.__name__}")

        try:
            explainer = self._xai_factory.create_explainer(
                name=explainer_name,
                model=self._model,
                use_defaults=use_defaults,
                **config_kwargs,
            )
            logger.info(
                f"Successfully created explainer: {explainer.__class__.__name__}"
            )
            logger.info(f"Explainer name method: {explainer.get_name()}")
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

    def _log_comprehensive_config(self) -> None:
        """
        Logs comprehensive configuration parameters to MLflow.
        """
        # Model configuration
        mlflow.log_param("model_name", self._config.model.name)
        mlflow.log_param("model_pretrained", self._config.model.pretrained)
        mlflow.log_param("model_transform", self._config.model.transform)

        # Explainer configuration
        mlflow.log_param("explainer_name", self._config.xai.name)
        mlflow.log_param("explainer_use_defaults", self._config.xai.use_defaults)
        for key, value in self._config.xai.kwargs.items():
            mlflow.log_param(f"explainer_{key}", value)

        # Data configuration
        mlflow.log_param("batch_size", self._config.data.batch_size)
        mlflow.log_param("max_batches", self._config.data.max_batches)
        mlflow.log_param("num_workers", self._config.data.num_workers)
        mlflow.log_param("shuffle", self._config.data.shuffle)
        mlflow.log_param("pin_memory", self._config.data.pin_memory)
        mlflow.log_param("resize", str(self._config.data.resize))

        # Hardware configuration
        mlflow.log_param("use_cuda", self._config.hardware.use_cuda)
        mlflow.log_param("device", self._config.hardware.device)

        # Metric configuration
        for metric_name, metric_config in self._config.metric.kwargs.items():
            for key, value in metric_config.items():
                mlflow.log_param(f"metric_{metric_name}_{key}", value)

        # Visualization configuration
        mlflow.log_param("visualization_save", self._config.visualization.save)
        mlflow.log_param("visualization_show", self._config.visualization.show)
        mlflow.log_param(
            "max_visualizations", self._config.visualization.max_visualizations
        )

        # Experiment configuration
        mlflow.log_param("top_k", self._config.experiment.top_k)
        mlflow.log_param("output_dir", self._config.experiment.output_dir)
        mlflow.log_param("seed", self._config.experiment.seed)

        # Logging configuration
        mlflow.log_param("logging_level", self._config.logging.level)

        # MLflow configuration parameters
        mlflow.log_param("mlflow_tracking_uri", self._config.mlflow.tracking_uri)
        mlflow.log_param("mlflow_experiment_name", self._config.mlflow.experiment_name)
        mlflow.log_param("mlflow_auto_log", self._config.mlflow.auto_log)
        if self._config.mlflow.run_name:
            mlflow.log_param("mlflow_run_name", self._config.mlflow.run_name)

    def _log_model_conditionally(self) -> None:
        """
        Logs the model to MLflow only if it's not already registered or if explicitly requested.
        Prevents creating unnecessary model versions for the same model.
        """
        try:
            client = MlflowClient()

            # Check if model is already registered
            try:
                model_versions = client.search_model_versions(
                    f"name='{self._config.model.name}'"
                )
                if model_versions:
                    self._logger.info(
                        f"Model '{self._config.model.name}' already registered with"
                        f" {len(model_versions)} version(s)"
                    )
                    self._logger.info(
                        "Skipping model logging to avoid duplicate versions"
                    )
                    return
            except Exception:
                # Model doesn't exist yet, proceed with logging
                pass

            # Log model only if not already registered
            self._logger.info(f"Logging new model: {self._config.model.name}")
            mlflow.pytorch.log_model(
                self._model.pytorch_model,
                name="model",
                registered_model_name=self._config.model.name,
            )
            self._logger.info(
                f"Model '{self._config.model.name}' successfully logged to MLflow"
            )

        except Exception as e:
            self._logger.warning(f"Failed to log model conditionally: {e}")
            self._logger.info("Proceeding without model logging")
