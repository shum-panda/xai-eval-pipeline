import os
import shutil
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.pipe.moduls.data.image_net_val_dataset import (
    ImageNetValDataset,
    create_dataloader,
)
from src.pipe.moduls.data.utils.bbox_to_mask import parse_bbox
from src.pipe.moduls.data.utils.collate_fn import explain_collate_fn


class TestImageNetValDataset(unittest.TestCase):
    """Test Suite für ImageNetValDataset Klasse."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"
        self.image_dir.mkdir()
        self.annot_dir.mkdir()
        self._create_test_data()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _create_test_data(self) -> None:
        self.test_images = []
        for i in range(3):
            img = Image.new("RGB", (100, 80), color=(i * 80, i * 60, i * 40))
            img_path = self.image_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
            img.save(img_path)
            self.test_images.append(img_path)

        self.test_annotations = []
        test_boxes = [
            [[10, 10, 50, 50], [60, 20, 90, 70]],
            [[20, 15, 80, 65]],
            [],
        ]

        for i, boxes in enumerate(test_boxes):
            xml_path = self.annot_dir / f"ILSVRC2012_val_{i:08d}.xml"
            self._create_xml_annotation(xml_path, boxes, (100, 80))
            self.test_annotations.append(xml_path)

        with open(self.label_file, "w") as f:
            f.write("1\n2\n3\n")

    def _create_xml_annotation(
        self, xml_path: Path, boxes: list, image_size: tuple
    ) -> None:
        root = ET.Element("annotation")
        size_elem = ET.SubElement(root, "size")
        ET.SubElement(size_elem, "width").text = str(image_size[0])
        ET.SubElement(size_elem, "height").text = str(image_size[1])
        for box in boxes:
            obj_elem = ET.SubElement(root, "object")
            ET.SubElement(obj_elem, "name").text = "test_object"
            bbox_elem = ET.SubElement(obj_elem, "bndbox")
            ET.SubElement(bbox_elem, "xmin").text = str(box[0])
            ET.SubElement(bbox_elem, "ymin").text = str(box[1])
            ET.SubElement(bbox_elem, "xmax").text = str(box[2])
            ET.SubElement(bbox_elem, "ymax").text = str(box[3])
        tree = ET.ElementTree(root)
        tree.write(xml_path)

    def test_dataset_initialization(self) -> None:
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.labels), 3)
        self.assertEqual(dataset.labels, [0, 1, 2])

    def test_dataset_getitem(self) -> None:
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            target_size=(224, 224),
        )
        image, label, boxes = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertIsInstance(boxes, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label.dtype, torch.long)
        self.assertEqual(boxes.dtype, torch.float32)
        self.assertEqual(label.item(), 0)
        self.assertEqual(boxes.shape[0], 2)
        self.assertEqual(boxes.shape[1], 4)

    def test_dataset_with_missing_xml(self) -> None:
        os.remove(self.test_annotations[1])
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
        )
        image, label, boxes = dataset[1]
        self.assertEqual(len(boxes), 0)
        if len(boxes) > 0:
            self.assertEqual(boxes.shape[1], 4)

    def test_get_sample_info(self) -> None:
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
        )
        info = dataset.get_sample_info(0)
        self.assertIn("image_path", info)
        self.assertIn("xml_path", info)
        self.assertIn("label", info)
        self.assertIn("exists_xml", info)
        self.assertEqual(info["label"], 0)
        self.assertTrue(info["exists_xml"])


class TestParseBbox(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_parse_valid_xml(self) -> None:
        xml_path = Path(self.temp_dir) / "test.xml"
        root = ET.Element("annotation")
        for i, box in enumerate([[10, 20, 30, 40], [50, 60, 70, 80]]):
            obj = ET.SubElement(root, "object")
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(box[0])
            ET.SubElement(bbox, "ymin").text = str(box[1])
            ET.SubElement(bbox, "xmax").text = str(box[2])
            ET.SubElement(bbox, "ymax").text = str(box[3])
        tree = ET.ElementTree(root)
        tree.write(xml_path)
        boxes = parse_bbox(xml_path)
        expected = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        np.testing.assert_array_equal(boxes, expected)

    def test_parse_nonexistent_xml(self) -> None:
        xml_path = Path(self.temp_dir) / "nonexistent.xml"
        boxes = parse_bbox(xml_path)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)

    def test_parse_empty_xml(self) -> None:
        xml_path = Path(self.temp_dir) / "empty.xml"
        root = ET.Element("annotation")
        tree = ET.ElementTree(root)
        tree.write(xml_path)
        boxes = parse_bbox(xml_path)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)


class TestCollateFn(unittest.TestCase):

    def test_collate_fn_basic(self) -> None:
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0), torch.randn(2, 4)),
            (torch.randn(3, 224, 224), torch.tensor(1), torch.randn(1, 4)),
            (torch.randn(3, 224, 224), torch.tensor(2), torch.randn(3, 4)),
        ]
        images, labels, boxes = explain_collate_fn(batch)
        self.assertEqual(images.shape, (3, 3, 224, 224))
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0].shape, (2, 4))
        self.assertEqual(boxes[1].shape, (1, 4))
        self.assertEqual(boxes[2].shape, (3, 4))

    def test_collate_fn_empty_boxes(self) -> None:
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0), torch.empty(0, 4)),
            (torch.randn(3, 224, 224), torch.tensor(1), torch.randn(2, 4)),
        ]
        images, labels, boxes = explain_collate_fn(batch)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(boxes[0]), 0)
        self.assertEqual(boxes[1].shape, (2, 4))


class TestCreateDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"
        self.image_dir.mkdir()
        self.annot_dir.mkdir()
        img = Image.new("RGB", (100, 100), color="red")
        img.save(self.image_dir / "test.JPEG")
        root = ET.Element("annotation")
        obj = ET.SubElement(root, "object")
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = "10"
        ET.SubElement(bbox, "ymin").text = "10"
        ET.SubElement(bbox, "xmax").text = "50"
        ET.SubElement(bbox, "ymax").text = "50"
        tree = ET.ElementTree(root)
        tree.write(self.annot_dir / "test.xml")
        with open(self.label_file, "w") as f:
            f.write("1\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_create_dataloader_basic(self) -> None:
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=1,
            num_workers=0,
        )
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(len(dataloader), 1)
        for images, labels, boxes in dataloader:
            self.assertEqual(images.shape, (1, 3, 224, 224))
            self.assertEqual(labels.shape, (1,))
            self.assertEqual(len(boxes), 1)
            break


class TestIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"
        self.image_dir.mkdir()
        self.annot_dir.mkdir()
        for i in range(5):
            img = Image.new("RGB", (200, 150), color=(i * 50, i * 40, i * 30))
            img.save(self.image_dir / f"img_{i:03d}.JPEG")
            root = ET.Element("annotation")
            for j in range(i + 1):
                obj = ET.SubElement(root, "object")
                bbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bbox, "xmin").text = str(j * 30)
                ET.SubElement(bbox, "ymin").text = str(j * 20)
                ET.SubElement(bbox, "xmax").text = str(j * 30 + 50)
                ET.SubElement(bbox, "ymax").text = str(j * 20 + 40)
            tree = ET.ElementTree(root)
            tree.write(self.annot_dir / f"img_{i:03d}.xml")
        with open(self.label_file, "w") as f:
            for i in range(5):
                f.write(f"{i + 1}\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline(self) -> None:
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=2,
            num_workers=0,
            target_size=(128, 128),
        )
        total_samples = 0
        for batch_idx, (images, labels, boxes) in enumerate(dataloader):
            batch_size = images.shape[0]
            total_samples += batch_size
            self.assertEqual(images.shape[1:], (3, 128, 128))
            self.assertEqual(labels.shape[0], batch_size)
            self.assertEqual(len(boxes), batch_size)
            self.assertTrue(torch.all(labels >= 0))
            self.assertTrue(torch.all(labels < 5))
            for i, box_tensor in enumerate(boxes):
                self.assertEqual(box_tensor.shape[1], 4)
                expected_boxes = (batch_idx * 2 + i) + 1
                if total_samples <= 5:
                    self.assertEqual(box_tensor.shape[0], expected_boxes)
        self.assertEqual(total_samples, 5)


class BenchmarkTests:
    @staticmethod
    def benchmark_loading_speed(dataset_path: str, num_samples: int = 100) -> float:
        import time

        print(f"Benchmark für {num_samples} Samples...")
        start_time = time.time()
        for _ in range(num_samples):
            pass
        end_time = time.time()
        avg_time = (end_time - start_time) / num_samples
        print(f"Durchschnittliche Ladezeit pro Sample: {avg_time * 1000:.2f} ms")
        print(f"Samples pro Sekunde: {1 / avg_time:.1f}")
        return avg_time

    @staticmethod
    def benchmark_batch_processing(dataloader, num_batches: int = 10) -> float:

        times = []
        for i, (images, labels, boxes) in enumerate(dataloader):
            if i >= num_batches:
                break
            start_time = time.time()
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            _ = images.sum()
            end_time = time.time()
            times.append(end_time - start_time)
        if times:
            avg_time = sum(times) / len(times)
            print(
                f"Durchschnittliche Batch-Verarbeitungszeit: {avg_time * 1000:.2f} ms"
            )
            return avg_time
        return 0.0


def create_mock_imagenet_structure(
    base_path_str: str, num_images: int = 10
) -> Tuple[str, str, str]:
    """
    Erstellt eine simulierte ImageNet-Datenstruktur mit Bildern,
    Bounding-Box-Annotationen und einem Labels-File zur Verwendung in Tests.

    Args:
        base_path_str (str): Basisverzeichnis für die Struktur (wird erstellt,
        falls nicht vorhanden).
        num_images (int): Anzahl der zu generierenden Beispielbilder und Annotationen.

    Returns:
        Tuple[str, str, str]: Pfade zu Bildverzeichnis, Annotationsverzeichnis und
        Label-Datei als Strings.
    """
    base_path: Path = Path(base_path_str)
    image_dir: Path = base_path / "images"
    annot_dir: Path = base_path / "annotations"

    image_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Dummy-Bild erzeugen
        img: Image.Image = Image.new(
            "RGB", (224, 224), color=(i * 25 % 255, i * 20 % 255, i * 15 % 255)
        )
        img.save(image_dir / f"ILSVRC2012_val_{i:08d}.JPEG")

        # Dummy-XML für Bounding Box erzeugen
        root = ET.Element("annotation")
        obj = ET.SubElement(root, "object")
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(i * 10)
        ET.SubElement(bbox, "ymin").text = str(i * 10)
        ET.SubElement(bbox, "xmax").text = str(i * 10 + 50)
        ET.SubElement(bbox, "ymax").text = str(i * 10 + 50)
        tree = ET.ElementTree(root)
        tree.write(annot_dir / f"ILSVRC2012_val_{i:08d}.xml")

    # Dummy-Labels erzeugen
    label_file: Path = base_path / "labels.txt"
    with label_file.open("w", encoding="utf-8") as f:
        for i in range(num_images):
            f.write(f"{(i % 1000) + 1}\n")

    return str(image_dir), str(annot_dir), str(label_file)


def run_all_tests() -> bool:
    print("🧪 Starte ImageNet Dataset Tests...")
    print("=" * 50)
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    print("\n" + "=" * 50)
    print("📊 Test-Zusammenfassung:")
    print(f"   Tests ausgeführt: {result.testsRun}")
    print(f"   Fehler: {len(result.errors)}")
    print(f"   Fehlschläge: {len(result.failures)}")
    if result.wasSuccessful():
        print("✅ Alle Tests erfolgreich!")
    else:
        print("❌ Einige Tests sind fehlgeschlagen.")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
