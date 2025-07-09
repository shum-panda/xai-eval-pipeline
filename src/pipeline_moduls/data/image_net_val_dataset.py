import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.transform import resize_boxes

from pipeline_moduls.data.dataclass.image_net_sample import ImageNetSample
from pipeline_moduls.data.utils.auto_batchsize_test import auto_batchsize_test
from pipeline_moduls.data.utils.bbox_to_mask import parse_bbox
from pipeline_moduls.data.utils.collate_fn import explain_collate_fn


class ImageNetValDataset(Dataset):
    """
    Effizientes Dataset für ImageNet Validation Daten mit Bounding Boxes.

    Args:
        image_dir: Pfad zu den .JPEG Bildern
        annot_dir: Pfad zu den .xml Annotationen
        label_file: Pfad zur Groundtruth-Datei (ILSVRC2012_validation_ground_truth.txt)
        transform: Optional Bildtransformationen
        target_size: Optional Zielgröße für Bilder (tuple: (H, W))
        cache_annotations: Ob XML-Annotationen gecacht werden sollen
    """

    def __init__(
        self,
        image_dir: Path,
        annot_dir: Path,
        label_file: Path,
        transform: Optional[transforms.Compose] = None,
        target_size: Optional[Tuple[int, int]] = None,
        cache_annotations: bool = True,
    ):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.target_size = target_size
        self.cache_annotations = cache_annotations

        # Lade Groundtruth Labels
        self.labels = self._load_labels(label_file)

        # Finde alle verfügbaren Bilder
        self.image_files = sorted([f for f in self.image_dir.glob("*.JPEG")])

        # Überprüfe Konsistenz
        if len(self.image_files) != len(self.labels):
            warnings.warn(
                f"Anzahl Bilder ({len(self.image_files)}) != Anzahl Labels "
                f"({len(self.labels)})"
            )

        # Standard Transform falls nicht angegeben
        if transform is None:
            if target_size:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(target_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transform = transform

        # Cache für Annotationen
        self._annotation_cache = {} if cache_annotations else None

        print("ImageNet Validation Dataset initialized:")
        print(f"  - {len(self.image_files)} images found")
        print(f"  - {len(self.labels)} labels loaded")
        print(f"  - Target size: {target_size}")
        print(f"  - Annotation caching: {cache_annotations}")

    def _load_labels(self, label_file: Path) -> List[int]:
        """
        Lädt die Groundtruth Labels aus der Textdatei.

        Die Labels in der Datei sind 1-indexiert (z. B. aus ImageNet),
        werden hier jedoch zu 0-indexiert konvertiert, um mit PyTorch-Modellen kompatibel zu sein.

        Args:
            label_file (Path): Pfad zur Textdatei mit den Ground-truth-Labels

        Returns:
            List[int]: Liste von 0-indexierten Klassenlabels
        """
        labels = []
        with label_file.open("r") as f:
            for line in f:
                # Labels sind 1-indexiert, konvertiere zu 0-indexiert
                label = int(line.strip()) - 1
                labels.append(label)
        return labels

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> ImageNetSample:
        """
        Lädt ein Sample bestehend aus Bild, Label und Bounding Boxes.

        Returns:
            image: Tensor [3, H, W]
            label: Tensor (long)
            boxes: Tensor [N, 4] mit Format [xmin, ymin, xmax, ymax]
        """
        image_path = self.image_files[idx]
        label = self.labels[idx]
        # Lade Bild
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (W, H)

        # Lade Bounding Boxes
        xml_path = self.annot_dir / f"{image_path.stem}.xml"
        boxes = self._get_boxes(xml_path, orig_size)

        # Transformiere Bild
        if self.transform:
            image = self.transform(image)

        # Konvertiere zu Tensoren
        label_tensor = torch.tensor(label, dtype=torch.long)
        boxes_tensor = boxes.clone().detach().float()

        # Dataclass befüllen und zurückgeben
        sample = ImageNetSample(
            image_name=image_path.name,
            image_path=image_path,
            image_tensor=image,
            label=label,
            label_tensor=label_tensor,
            bbox_path=xml_path,
            bbox_tensor=boxes_tensor,
        )
        return sample

    def _get_boxes(self, xml_path: Path, orig_size: Tuple[int, int]) -> torch.Tensor:
        """Lädt und verarbeitet Bounding Boxes aus XML-Datei."""
        # Prüfe Cache
        if (
            self._annotation_cache is not None
            and str(xml_path) in self._annotation_cache
        ):
            boxes = self._annotation_cache[str(xml_path)]
        else:
            try:
                boxes = parse_bbox(xml_path)
                if self._annotation_cache is not None:
                    self._annotation_cache[str(xml_path)] = boxes
            except FileNotFoundError:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                if self._annotation_cache is not None:
                    self._annotation_cache[str(xml_path)] = boxes

        # Skaliere Bounding Boxes falls nötig
        if self.target_size and boxes.numel() > 0:
            orig_h, orig_w = orig_size
            new_h, new_w = self.target_size
            boxes = resize_boxes(boxes, [orig_h, orig_w], [new_h, new_w])

        return boxes

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Gibt zusätzliche Informationen über ein Sample zurück."""
        image_path = self.image_files[idx]
        xml_path = self.annot_dir / f"{image_path.stem}.xml"

        return {
            "image_path": str(image_path),
            "xml_path": str(xml_path),
            "label": self.labels[idx],
            "exists_xml": xml_path.exists(),
        }


def create_dataloader(
    image_dir: Path,
    annot_dir: Path,
    label_file: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = False,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    transform: Optional[transforms.Compose] = None,
    custom_collate_fn=explain_collate_fn,
) -> DataLoader:
    """
    Erstellt einen konfigurierten DataLoader für ImageNet Validation.

    Args:
        image_dir: Pfad zu den Bildern
        annot_dir: Pfad zu den Annotationen
        label_file: Pfad zur Groundtruth-Datei
        batch_size: Batch-Größe
        num_workers: Anzahl Worker-Prozesse
        pin_memory: Ob Speicher gepinnt werden soll (für GPU)
        shuffle: Ob Daten gemischt werden sollen
        target_size: Zielgröße für Bilder
        transform: Optional custom Transform
        custom_collate_fn: Collate_fn for the dataset

    Returns:
        DataLoader: Konfigurierter DataLoader
    """
    dataset = ImageNetValDataset(
        image_dir=image_dir,
        annot_dir=annot_dir,
        label_file=label_file,
        transform=transform,
        target_size=target_size,
    )

    image_net_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )

    return image_net_dataloader


# Beispiel für Verwendung
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    image_dir = project_root / "data" / "extracted" / "validation_images"
    annot_dir = project_root / "data" / "extracted" / "bounding_boxes"
    label_file = project_root / "data" / "ILSVRC2012_validation_ground_truth.txt"

    # Erstelle DataLoader
    dataloader = create_dataloader(
        image_dir=image_dir,
        annot_dir=annot_dir,
        label_file=label_file,
        batch_size=16,
        num_workers=4,
        target_size=(224, 224),
    )

    print(f"DataLoader erstellt mit {len(dataloader)} Batches")

    # Teste ersten Batch
    for images, labels, boxes in dataloader:
        print(f"Batch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Boxes: {len(boxes)} samples, shapes: {[b.shape for b in boxes[:3]]}")
        break

    # Optional: Teste optimale Batch-Größe
    optimal_batch_size = auto_batchsize_test(dataloader)
    print(optimal_batch_size)
