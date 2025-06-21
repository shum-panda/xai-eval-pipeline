
"""
XAI Orchestrator - Koordiniert Models, Explainer und Dataset
Getrennt von der Evaluation Logic
"""

import logging
import time

import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass
import sys

from torch.utils.data import DataLoader
from torchvision import transforms

from data.image_net_val_dataset import ImageNetValDataset, collate_fn
from models.model_factory import ModelFactory
from xai_methods.base.base_explainer import BaseExplainer
from xai_methods.xai_factory import XAIFactory

# Import ImageNet Dataset
sys.path.append(str(Path(__file__).parent.parent / "data"))


@dataclass
class XAIExplanationResult:
    """
    Strukturiertes Ergebnis einer XAI Explanation
    """
    image:torch.Tensor
    image_name: str

    # Model Prediction
    predicted_class: int
    true_label: Optional[int]
    prediction_correct: Optional[bool]

    # XAI Explanation
    attribution: torch.Tensor
    explainer_result: Any  # Dein ExplainerResult oder ähnlich
    explainer_name: str

    # Dataset Info
    has_bbox: bool
    bbox: torch.Tensor
    bbox_info: Optional[Dict]


    # Metadata
    model_name: str
    processing_time: float


class XAIOrchestrator:
    """
    Orchestriert XAI Pipeline - Model, Dataset, Explainer
    verantwortlich nur für die Koordination, nicht für Evaluation
    """

    def __init__(self, model_name: str = "resnet50"):
        """
        Initialisiere Orchestrator

        Args:
            model_name: Name des Models für ModelFactory
        """
        self.logger = logging.getLogger(__name__)

        # Setup Factories
        self.model_factory = ModelFactory.get_instance()
        self.xai_factory = XAIFactory()

        # Load Model
        self.model_interface = self.model_factory.load_model(model_name)
        self.pytorch_model = self.model_interface.get_pytorch_model()
        self.device = next(self.pytorch_model.parameters()).device
        self.model_name = model_name

        self.logger.info(f"Orchestrator initialisiert:")
        self.logger.info(f"  Model: {model_name} auf {self.device}")
        self.logger.info(f"  Verfügbare Explainer: {self.xai_factory.list_available_explainers()}")

    def setup_dataloader(
            self,
            project_root: Optional[Path] = None,
            batch_size: int = 16,
            num_workers: int = 4,
            pin_memory: bool = True,
            shuffle: bool = False,
            target_size: Optional[Tuple[int, int]] = (224, 224),
            transform: Optional[transforms.Compose] = None
    ) -> torch.utils.data.DataLoader:
        """
        Setup ImageNet DataLoader mit konfigurierbarem Pfad und Einstellungen.

        Args:
            project_root: Projektwurzel (optional, sonst automatisch erkannt)
            batch_size: Batch-Größe für das Laden
            num_workers: Worker für paralleles Laden
            pin_memory: Optimierung für GPU
            shuffle: Shuffle-Option
            target_size: Zielgröße für Bilder
            transform: Custom Transform

        Returns:
            DataLoader: Konfigurierter DataLoader
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        image_dir = project_root / "data" / "extracted" / "validation_images"
        annot_dir = project_root / "data" / "extracted" / "bounding_boxes"
        label_file = project_root / "data" / "ILSVRC2012_validation_ground_truth.txt"

        dataset = ImageNetValDataset(
            image_dir=image_dir,
            annot_dir=annot_dir,
            label_file=label_file,
            target_size=target_size,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

        self.logger.info(f"DataLoader setup: {len(dataset)} Samples in {len(dataloader)} Batches")
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
            batch: Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
            explainer: BaseExplainer,
            image_names: Optional[List[str]] = None
    ) -> List[XAIExplanationResult]:
        """
        Erklärt einen Batch von Bildern mit XAI-Explainer.

        Args:
            batch: Tuple (images, labels, boxes)
            explainer: Konfigurierter Explainer
            image_names: Optional Liste von Bildnamen

        Returns:
            Liste von XAIExplanationResult
        """
        start_time = time.time()

        images, labels, boxes_list = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Run XAI explanation (inkl. Prediction!)
        explanation_result=None
        try:
            explanation_result = explainer.explain(images, labels)
            attributions = explanation_result.attributions  # [B, C, H, W]
            pred_classes = explanation_result.predictions  # [B]
            target_labels = explanation_result.target_labels  # [B]

        except Exception as e:
            self.logger.error(f"Fehler beim Erklären eines Batches: {e}")
            B = images.size(0)
            attributions = torch.zeros_like(images)
            pred_classes = torch.full((B,), -1)
            target_labels = labels

        processing_time = time.time() - start_time

        results = []
        for i in range(images.size(0)):
            result = XAIExplanationResult(
                image=images[i],
                image_name=image_names[i] if image_names else f"image_{i}",
                predicted_class=pred_classes[i].argmax().item(),
                true_label=int(target_labels[i]),
                prediction_correct=int(pred_classes[i].argmax().item()) == int(target_labels[i]),
                attribution=attributions[i].detach().cpu(),
                explainer_result=explanation_result,
                explainer_name=explainer.__class__.__name__,
                has_bbox=boxes_list[i].numel() > 0,
                bbox=boxes_list[i],
                bbox_info=boxes_list[i].detach().cpu(),
                model_name=self.model_name,
                processing_time=processing_time / images.size(0)
            )
            results.append(result)

        return results

    def batch_process(self, dataloader : DataLoader,
                      explainer_names: List[str],
                      explainer_kwargs: Optional[Dict[str, Dict]] = None) -> Dict[str, List[XAIExplanationResult]]:
        """
        Batch Verarbeitung mit mehreren Explainern

        Args:
            dataloader: ImageNet Dataloader
            explainer_names: Liste der Explainer Namen
            max_samples: Max Anzahl Samples
            explainer_kwargs: Kwargs für jeden Explainer

        Returns:
            Results gruppiert nach Explainer
            :param dataloader:
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
                    explainer=explainer
                ))

                results[explainer_name] = explainer_results

            except Exception as e:
                self.logger.error(f"Fehler bei {explainer_name}: {e}")
                results[explainer_name] = []

        return results

    def process_dataloader(
            self,
            dataloader: torch.utils.data.DataLoader,
            explainer: BaseExplainer,
            max_batches:Optional[int]=None
    ) -> Iterator[XAIExplanationResult]:
        """
        Verarbeite Dataset batchweise mit Explainer (Generator für Memory-Effizienz)

        Args:
            dataloader: DataLoader mit Batch-Samples (images, labels, boxes)
            explainer: Konfigurierter Explainer
            max_batches: Maximale Anzahl an Batches
        Yields:
            XAIExplanationResult-Objekte pro Bild
        """
        total_batches = len(dataloader)
        self.logger.info(f"Starte Verarbeitung von {total_batches} Batches...")

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:

                # Erklärung für den Batch erzeugen
                results = self.explain_batch(batch, explainer)

                for res in results:
                    yield res

                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Fortschritt: {batch_idx + 1}/{total_batches} Batches")

            except Exception as e:
                self.logger.error(f"Fehler bei Batch {batch_idx}: {e}")
                continue

    def switch_model(self, model_name: str):
        """
        Wechsle zu anderem Model

        Args:
            model_name: Neuer Model Name
        """
        self.model_interface = self.model_factory.load_model(model_name)
        self.pytorch_model = self.model_interface.get_pytorch_model()
        self.device = next(self.pytorch_model.parameters()).device
        self.model_name = model_name

        self.logger.info(f"Model gewechselt zu: {model_name}")

    def get_available_explainers(self) -> List[str]:
        """Hole verfügbare Explainer"""
        return self.xai_factory.list_available_explainers()

    def get_model_info(self) -> Dict[str, Any]:
        """Hole Model Informationen"""
        return self.model_interface.get_model_info()

    def _extract_attribution(self, explanation_result_for_one_image):
        """ gets the attributions for givven XAI Methode"""


def main():
    """Beispiel Usage des Orchestrators mit Batch-Verarbeitung"""
    logging.basicConfig(level=logging.INFO)

    print("🎼 XAI Orchestrator")
    print("=" * 50)

    try:
        # 1. Setup Orchestrator
        orchestrator = XAIOrchestrator("resnet50")

        # 2. Setup DataLoader (Batchweise Daten)
        dataloader = orchestrator.setup_dataloader()

        # 3. Info
        print(f"\n📊 Setup:")
        print(f"  Model: {orchestrator.get_model_info()}")
        print(f"  DataLoader mit {len(dataloader)} Items, Batch-Größe {dataloader.batch_size}")
        print(f"  Explainer: {orchestrator.get_available_explainers()}")

        # 4. Batchweises Testen mit erstem Explainer
        if orchestrator.get_available_explainers():
            explainer_name = orchestrator.get_available_explainers()[0]
            explainer = orchestrator.create_explainer(explainer_name)

            print(f"\n🔬 Teste {explainer_name} batchweise...")

            # Erster Batch (mit Bildern, Labels, Bounding Boxes)
            batch = next(iter(dataloader))
            images, labels, boxes = batch

            # Erkläre Batch (angenommen explainer.explain() unterstützt batch)
            explanation_result = orchestrator.explain_batch(batch, explainer)

            # Beispiel-Ausgabe für das erste Bild im Batch
            attribution = explanation_result[0].attribution
            pred_class = explanation_result[0].predicted_class

            print(f"✅ Ergebnis erster Batch:")
            print(f"  Anzahl Bilder: {images.size(0)}")
            print(f"  Erste Prediction: {pred_class}")
            print(f"  Erste Ground Truth: {labels[0].item()}")
            print(f"  Attribution Shape (erstes Bild): {attribution[0].shape if hasattr(attribution, '__len__') else 'N/A'}")
            print(explanation_result[2])


        print(f"\n✅ Orchestrator Batch-Test abgeschlossen!")

    except Exception as e:
        logging.error(f"Orchestrator Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()