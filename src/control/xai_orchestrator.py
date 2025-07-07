
"""
XAI Orchestrator - Koordiniert Models, Explainer und Dataset
Getrennt von der Evaluation Logic
"""

import logging
import time

import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple, Union
import sys

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from control.utils.config_dataclasses.master_config import MasterConfig
from control.utils.dataclasses.xai_explanation_result import XAIExplanationResult
from control.utils.describe_batch import describe_batch
from pipeline_moduls.ResultManager.result_manager import ResultManager
from pipeline_moduls.data.image_net_val_dataset import create_dataloader
from pipeline_moduls.data.utils.collate_fn import explain_collate_fn
from pipeline_moduls.evaluation.metric_calculator import XAIEvaluator
from pipeline_moduls.models.base.interface.xai_model import XAIModel
from pipeline_moduls.models.base.xai_model_factory import XAIModelFactory
from pipeline_moduls.visualization.visualisation import Visualiser
from pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from pipeline_moduls.xai_methods.xai_factory import XAIFactory

# Import ImageNet Dataset
sys.path.append(str(Path(__file__).parent.parent / "data"))



class XAIOrchestrator:
    """
    Orchestriert XAI Pipeline - Model, Dataset, Explainer
    verantwortlich nur für die Koordination, nicht für Evaluation
    """

    def __init__(self, config: MasterConfig):
        """
        Initializes the XAI Orchestrator with the given configuration.

        Args:
            config: MasterConfig object loaded by Hydra.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.result_manager = ResultManager()

        # Setup factories
        self.model_factory: XAIModelFactory = XAIModelFactory()
        self.xai_factory: XAIFactory = XAIFactory()

        # Load model
        self.model_name: str = self.config.model.name
        self.model: XAIModel = self.model_factory.create(self.model_name)
        self.pytorch_model = self.model.get_pytorch_model()
        self.device = next(self.pytorch_model.parameters()).device

        # Evaluator and Visualizer
        self.evaluator = XAIEvaluator()
        self.visualiser = Visualiser(
            show=False,  # Replace with self.config.visualization.show if you add that field
            save_path=Path(self.config.experiment.output_dir)
        )

        self.logger.info(f"Orchestrator initialized:")
        self.logger.info(f"  Model: {self.model_name} on {self.device}")
        self.logger.info(f"  Available explainers: {self.xai_factory.list_available_explainers()}")

    def setup_dataloader(
            self,
            project_root: Optional[Path] = None,
            batch_size: int = 16,
            num_workers: int = 4,
            pin_memory: bool = True,
            shuffle: bool = False,
            target_size: Optional[Tuple[int, int]] = (224, 224),
            transform: Optional[transforms.Compose] = None
    ) -> DataLoader:
        """
        Sets up the ImageNet DataLoader with configurable parameters.

        Args:
            project_root (Optional[Path]): Root path of the project. If None, it is inferred automatically.
            batch_size (int): Batch size for loading the dataset.
            num_workers (int): Number of subprocesses used for data loading.
            pin_memory (bool): Whether to pin memory (recommended for GPU).
            shuffle (bool): Whether to shuffle the dataset during loading.
            target_size (Optional[Tuple[int, int]]): Target size for resizing images (e.g., (224, 224)).
            transform (Optional[transforms.Compose]): Optional custom transform to apply to images.

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
        label_file = project_root / "data" / "ILSVRC2012_validation_ground_truth.txt"

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
            custom_collate_fn=explain_collate_fn
        )

        self.logger.info(
            f"DataLoader setup complete: {len(dataloader.dataset)} samples in {len(dataloader)} batches "
            f"(batch_size={batch_size}, shuffle={shuffle})"
        )

        return dataloader

    def create_explainer(self, explainer_name: str, **kwargs) -> BaseExplainer:
        """
        Erstelle Explainer über XAI Factory

        Args:
            explainer_name: Name des Explainers
            **kwargs: Zusätzliche Parameter

        Returns:
            Konfigurierter Explainer
        """
        explainer = self.xai_factory.create_explainer(
            name=explainer_name,
            model=self.pytorch_model,
            **kwargs
        )

        self.logger.info(f"Explainer '{explainer_name}' erstellt")
        return explainer

    def explain_batch(
            self,
            batch: Union[
                Tuple[
                    Tensor, Tensor, List[Tensor], List[Path], List[str], List[Path], List[int]
                ],
                object
            ],
            explainer: BaseExplainer,
    ) -> List[XAIExplanationResult]:
        """
        Explains a batch of images using the provided XAI explainer.

        Args:
            batch: Expected to be a tuple:
                (images_tensor, labels_tensor, boxes_list, image_paths, image_names, bbox_paths, labels_int)
            explainer: Configured explainer instance.

        Returns:
            List of XAIExplanationResult instances.

        Raises:
            TypeError: If batch does not match the expected structure.
        """
        # Simple sanity check of batch structure
        if (
                not isinstance(batch, tuple) or
                len(batch) != 7 or
                not isinstance(batch[0], torch.Tensor) or
                not isinstance(batch[1], torch.Tensor) or
                not isinstance(batch[2], list) or
                not all(isinstance(b, torch.Tensor) for b in batch[2]) or
                not isinstance(batch[3], list) or
                not all(isinstance(p, Path) for p in batch[3]) or
                not isinstance(batch[4], list) or
                not all(isinstance(n, str) for n in batch[4]) or
                not isinstance(batch[5], list) or
                not all(isinstance(p, Path) for p in batch[5]) or
                not isinstance(batch[6], list) or
                not all(isinstance(l, int) for l in batch[6])
        ):
            raise TypeError(
                "Batchtyp:", describe_batch(batch),
                "Input batch does not match expected structure:\n"
                "Expected tuple of (Tensor, Tensor, List[Tensor], List[Path], List[str], List[Path], List[int])."
            )
        start_time = time.time()

        images, labels_tensor, boxes_list, image_paths, image_names, bbox_paths, labels_int = batch
        images = images.to(self.device)
        labels_tensor = labels_tensor.to(self.device)

        explanation_result = None
        try:
            explanation_result = explainer.explain(images, labels_tensor)
            attributions = explanation_result.attributions  # [B, C, H, W]
            pred_classes = explanation_result.predictions  # e.g. [B] or [B, num_classes]
            target_labels = explanation_result.target_labels  # [B]

        except Exception as e:
            self.logger.error(f"Error explaining batch: {e}")
            B = images.size(0)
            attributions = torch.zeros_like(images)
            pred_classes = torch.full((B,), -1)
            target_labels = labels_tensor

        processing_time = time.time() - start_time

        results = []
        for i in range(images.size(0)):
            if pred_classes.dim() == 2:
                predicted_class = pred_classes[i].argmax().item()
            else:
                predicted_class = pred_classes[i].item()

            true_label = int(target_labels[i])
            image_name = image_names[i] if image_names else f"image_{i}"

            result = XAIExplanationResult(
                image=images[i],
                image_path=image_paths[i],
                image_name=image_name,
                predicted_class=predicted_class,
                true_label=true_label,
                prediction_correct=(predicted_class == true_label),
                attribution=attributions[i].detach().cpu(),
                explainer_result=explanation_result,
                explainer_name=explainer.__class__.__name__,
                has_bbox=boxes_list[i].numel() > 0,
                bbox=boxes_list[i],
                bbox_info=boxes_list[i].detach().cpu(),
                model_name=self.model_name,
                processing_time=processing_time / images.size(0),
            )
            results.append(result)

        return results

    def batch_process(self, dataloader : DataLoader,
                      explainer_names: List[str],
                      max_batches:Optional[int]=None,
                      explainer_kwargs: Optional[Dict[str, Dict]] = None) -> Dict[str, List[XAIExplanationResult]]:
        """
        Batch Verarbeitung mit mehreren Explainern

        Args:
            dataloader: ImageNet Dataloader
            explainer_names: Liste der Explainer Namen
            max_batches: Max Anzahl batches
            explainer_kwargs: Kwargs für jeden Explainer

        Returns:
            Results gruppiert nach Explainer
        """
        explainer_kwargs = explainer_kwargs or {}
        results = {}


        for explainer_name in explainer_names:
            self.logger.info(f"Verarbeite mit {explainer_name}...")

            try:
                # Erstelle Explainer
                kwargs = explainer_kwargs.get(explainer_name, {})
                explainer = self.create_explainer(explainer_name, **kwargs)

                # Verarbeite Dataset
                explainer_results = list(self.process_dataloader(
                    dataloader=dataloader,
                    explainer=explainer,
                    max_batches=max_batches
                ))
                self.result_manager.add_results(explainer_results)
                results[explainer_name] = explainer_results

            except Exception as e:
                self.logger.error(f"Fehler bei {explainer_name}: {e}")
                results[explainer_name] = []

        return results

    def process_dataloader(
            self,
            dataloader: DataLoader,
            explainer: BaseExplainer,
            max_batches:Optional[int]=None
    ) -> Iterator[XAIExplanationResult]:
        """
        Iterates over a DataLoader and explains each batch using a given XAI explainer.
        Yields XAIExplanationResult objects for each image in a memory-efficient way.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader that yields batches created by
                a custom `collate_fn`. The collate function must return a tuple of:
                (images_tensor, labels_tensor, boxes_list, image_paths, image_names, bbox_paths, labels_int).
            explainer (BaseExplainer): The configured XAI explainer used to generate attributions.
            max_batches (Optional[int]): Optional limit on the number of batches to process.

        Yields:
            XAIExplanationResult: One result per image in the batch.

        Notes:
            - The `explain_batch` function performs runtime type validation on the batch structure.
            - If the batch does not match the expected format, a TypeError will be raised.
            - This error is caught and logged, and processing continues with the next batch.
            - Other unexpected exceptions are also caught and logged.
        """
        total_batches = len(dataloader)
        self.logger.info(f"Starting processing of {total_batches} batches...")

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:
                results = self.explain_batch(tuple(batch), explainer)

                for res in results:
                    yield res

                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Progress: {batch_idx + 1}/{total_batches} batches")

            except TypeError as e:
                self.logger.error(f"Type error in batch {batch_idx}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error in batch {batch_idx}: {e}")
            finally:
                torch.cuda.empty_cache()

    def switch_model(self, model_name: str):
        """
        Wechsle zu anderem Model

        Args:
            model_name: Neuer Model Name
        """
        self.model = self.model_factory.create(model_name)
        self.pytorch_model = self.model.get_pytorch_model()
        self.device = next(self.pytorch_model.parameters()).device
        self.model_name = model_name

        self.logger.info(f"Model gewechselt zu: {model_name}")

    def get_available_explainers(self) -> List[str]:
        """Hole verfügbare Explainer"""
        return self.xai_factory.list_available_explainers()

    def get_model_info(self) -> Dict[str, Any]:
        """Hole Model Informationen"""
        return self.model.get_model_info()

    def _extract_attribution(self, explanation_result_for_one_image):
        """ gets the attributions for givven XAI Methode"""

    def run(self) -> Dict[str, Any]:
        """
        Main entry point for running the XAI pipeline.
        """
        self.logger.info(f"Starting experiment: {self.config.experiment.name}")

        # 1. Setup dataloader
        dataloader = self.setup_dataloader()

        # 2. Create explainer
        explainer = self.create_explainer(
            explainer_name=self.config.xai.name,
            **self.config.xai.kwargs
        )

        # 3. Process data and collect results
        self.logger.info(f"Processing samples with {self.config.xai.name}...")
        processed_samples = 0

        for _ in self.process_dataloader(
                dataloader=dataloader,
                explainer=explainer,
                max_batches=self.config.data.max_batches):
            processed_samples += 1

        self.logger.info(f"Processed {processed_samples} samples")

        # 4. Evaluate results
        results_df = self.result_manager.get_dataframe()  # Zugriff auf alle Ergebnisse
        self.logger.info("Calculating evaluation metrics...")
        summary = self.evaluator.evaluate_batch_results(results_df)

        # 5. Save results dataframe and summary
        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "results.csv"
        self.result_manager.save_dataframe(str(csv_path))
        self.logger.info(f"Results saved to {csv_path}")

        summary_path = output_dir / "metrics_summary.yaml"
        with open(summary_path, "w") as f:
            import yaml
            yaml.dump(summary, f)
        self.logger.info(f"Metrics summary saved to {summary_path}")

        # 6. Generate visualizations if enabled
        if self.config.visualization.save:
            self.logger.info("Generating visualizations...")
            for _, row in results_df.iterrows():
                result = XAIExplanationResult.from_dict(row.to_dict())
                metrics = self.evaluator.evaluate_single_result(result)
                self.visualiser.create_visualization(result=result, metrics=metrics)

        self.logger.info("Experiment complete!")

        return {
            'experiment_name': self.config.experiment.name,
            'explainer': self.config.xai.name,
            'total_samples': processed_samples,
            'summary': summary,
            'output_dir': str(output_dir),
            'results_csv': str(csv_path),
            'metrics_summary': str(summary_path)
        }

    def quick_test(self, n_samples: int = 5) -> Dict[str, Any]:
        """Schneller Test mit wenigen Samples"""
        original_max = self.config.data.max_samples
        original_viz = self.config.visualization.save

        # Temporär anpassen
        self.config.data.max_samples = n_samples
        self.config.visualization.save = False

        try:
            result = self.run()
            return result
        finally:
            # Restore
            self.config.data.max_samples = original_max
            self.config.visualization.save = original_viz

