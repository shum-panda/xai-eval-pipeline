import unittest
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import xml.etree.ElementTree as ET

from data.image_net_val_dataset import create_dataloader, collate_fn, parse_bbox, ImageNetValDataset


class TestImageNetValDataset(unittest.TestCase):
    """Test Suite für ImageNetValDataset Klasse."""

    def setUp(self):
        """Setup für jeden Test - erstellt temporäre Dateien und Verzeichnisse."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"

        # Erstelle Verzeichnisse
        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        # Erstelle Test-Daten
        self._create_test_data()

    def tearDown(self):
        """Cleanup nach jedem Test."""
        shutil.rmtree(self.temp_dir)

    def _create_test_data(self):
        """Erstellt Test-Bilder, Annotationen und Labels."""
        # Erstelle 3 Test-Bilder
        self.test_images = []
        for i in range(3):
            img = Image.new('RGB', (100, 80), color=(i * 80, i * 60, i * 40))
            img_path = self.image_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
            img.save(img_path)
            self.test_images.append(img_path)

        # Erstelle Test-Annotationen (XML)
        self.test_annotations = []
        test_boxes = [
            [[10, 10, 50, 50], [60, 20, 90, 70]],  # 2 Boxen
            [[20, 15, 80, 65]],  # 1 Box
            []  # Keine Boxen
        ]

        for i, boxes in enumerate(test_boxes):
            xml_path = self.annot_dir / f"ILSVRC2012_val_{i:08d}.xml"
            self._create_xml_annotation(xml_path, boxes, (100, 80))
            self.test_annotations.append(xml_path)

        # Erstelle Label-Datei (1-indexiert wie ImageNet)
        with open(self.label_file, 'w') as f:
            f.write("1\n2\n3\n")  # Labels für 3 Bilder

    def _create_xml_annotation(self, xml_path: Path, boxes: list, image_size: tuple):
        """Erstellt eine Pascal VOC XML-Annotation."""
        root = ET.Element("annotation")

        # Bildgröße
        size_elem = ET.SubElement(root, "size")
        ET.SubElement(size_elem, "width").text = str(image_size[0])
        ET.SubElement(size_elem, "height").text = str(image_size[1])

        # Objekte/Bounding Boxes
        for box in boxes:
            obj_elem = ET.SubElement(root, "object")
            ET.SubElement(obj_elem, "name").text = "test_object"

            bbox_elem = ET.SubElement(obj_elem, "bndbox")
            ET.SubElement(bbox_elem, "xmin").text = str(box[0])
            ET.SubElement(bbox_elem, "ymin").text = str(box[1])
            ET.SubElement(bbox_elem, "xmax").text = str(box[2])
            ET.SubElement(bbox_elem, "ymax").text = str(box[3])

        # Schreibe XML
        tree = ET.ElementTree(root)
        tree.write(xml_path)

    def test_dataset_initialization(self):
        """Test Dataset-Initialisierung."""
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file
        )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.labels), 3)
        self.assertEqual(dataset.labels, [0, 1, 2])  # 0-indexiert

    def test_dataset_getitem(self):
        """Test __getitem__ Methode."""
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            target_size=(224, 224)
        )

        # Test erstes Sample
        image, label, boxes = dataset[0]

        # Prüfe Tensor-Typen und -Formen
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertIsInstance(boxes, torch.Tensor)

        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label.dtype, torch.long)
        self.assertEqual(boxes.dtype, torch.float32)
        self.assertEqual(label.item(), 0)
        self.assertEqual(boxes.shape[0], 2)  # 2 Bounding Boxes
        self.assertEqual(boxes.shape[1], 4)  # [xmin, ymin, xmax, ymax]

    def test_dataset_with_missing_xml(self):
        """Test Verhalten bei fehlenden XML-Dateien."""
        # Lösche eine XML-Datei
        os.remove(self.test_annotations[1])

        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file
        )

        # Sample mit fehlender XML sollte leere Bounding Box Liste haben
        image, label, boxes = dataset[1]
        self.assertEqual(len(boxes), 0)
        if len(boxes) > 0:  # Nur prüfen wenn nicht leer
            self.assertEqual(boxes.shape[1], 4)

    def test_get_sample_info(self):
        """Test get_sample_info Methode."""
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file
        )

        info = dataset.get_sample_info(0)

        self.assertIn('image_path', info)
        self.assertIn('xml_path', info)
        self.assertIn('label', info)
        self.assertIn('exists_xml', info)

        self.assertEqual(info['label'], 0)
        self.assertTrue(info['exists_xml'])


class TestParseBbox(unittest.TestCase):
    """Test Suite für parse_bbox Funktion."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_parse_valid_xml(self):
        """Test Parsen einer gültigen XML-Datei."""
        xml_path = Path(self.temp_dir) / "test.xml"

        # Erstelle Test-XML
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

        # Parse XML
        boxes = parse_bbox(xml_path)

        expected = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        np.testing.assert_array_equal(boxes, expected)

    def test_parse_nonexistent_xml(self):
        """Test Verhalten bei nicht-existierender XML-Datei."""
        xml_path = Path(self.temp_dir) / "nonexistent.xml"
        boxes = parse_bbox(xml_path)

        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)

    def test_parse_empty_xml(self):
        """Test Parsen einer XML ohne Objekte."""
        xml_path = Path(self.temp_dir) / "empty.xml"

        root = ET.Element("annotation")
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        boxes = parse_bbox(xml_path)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)


class TestCollateFn(unittest.TestCase):
    """Test Suite für collate_fn."""

    def test_collate_fn_basic(self):
        """Test grundlegende Funktionalität der Collate Function."""
        # Simuliere Batch-Daten
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0), torch.randn(2, 4)),
            (torch.randn(3, 224, 224), torch.tensor(1), torch.randn(1, 4)),
            (torch.randn(3, 224, 224), torch.tensor(2), torch.randn(3, 4))
        ]

        images, labels, boxes = collate_fn(batch)

        # Prüfe Ausgabe-Formen
        self.assertEqual(images.shape, (3, 3, 224, 224))
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0].shape, (2, 4))
        self.assertEqual(boxes[1].shape, (1, 4))
        self.assertEqual(boxes[2].shape, (3, 4))

    def test_collate_fn_empty_boxes(self):
        """Test Collate Function mit leeren Bounding Boxes."""
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0), torch.empty(0, 4)),
            (torch.randn(3, 224, 224), torch.tensor(1), torch.randn(2, 4))
        ]

        images, labels, boxes = collate_fn(batch)

        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(boxes[0]), 0)  # Erste Box-Liste ist leer
        self.assertEqual(boxes[1].shape, (2, 4))


class TestCreateDataLoader(unittest.TestCase):
    """Test Suite für create_dataloader Funktion."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"

        # Erstelle Test-Daten
        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        # Erstelle ein Test-Bild und XML
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.image_dir / "test.JPEG")

        # Erstelle XML
        root = ET.Element("annotation")
        obj = ET.SubElement(root, "object")
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = "10"
        ET.SubElement(bbox, "ymin").text = "10"
        ET.SubElement(bbox, "xmax").text = "50"
        ET.SubElement(bbox, "ymax").text = "50"

        tree = ET.ElementTree(root)
        tree.write(self.annot_dir / "test.xml")

        # Erstelle Label-Datei
        with open(self.label_file, 'w') as f:
            f.write("1\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_dataloader_basic(self):
        """Test DataLoader-Erstellung."""
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=1,
            num_workers=0  # Keine Multiprocessing für Test
        )

        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(len(dataloader), 1)

        # Test einen Batch
        for images, labels, boxes in dataloader:
            self.assertEqual(images.shape, (1, 3, 224, 224))
            self.assertEqual(labels.shape, (1,))
            self.assertEqual(len(boxes), 1)
            break


class TestIntegration(unittest.TestCase):
    """Integrationstests für das gesamte Modul."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"

        # Erstelle umfangreichere Test-Daten
        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        # Erstelle 5 Test-Bilder
        for i in range(5):
            img = Image.new('RGB', (200, 150), color=(i * 50, i * 40, i * 30))
            img.save(self.image_dir / f"img_{i:03d}.JPEG")

            # Erstelle entsprechende XML
            root = ET.Element("annotation")
            for j in range(i + 1):  # Variable Anzahl Boxen
                obj = ET.SubElement(root, "object")
                bbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bbox, "xmin").text = str(j * 30)
                ET.SubElement(bbox, "ymin").text = str(j * 20)
                ET.SubElement(bbox, "xmax").text = str(j * 30 + 50)
                ET.SubElement(bbox, "ymax").text = str(j * 20 + 40)

            tree = ET.ElementTree(root)
            tree.write(self.annot_dir / f"img_{i:03d}.xml")

        # Erstelle Labels
        with open(self.label_file, 'w') as f:
            for i in range(5):
                f.write(f"{i + 1}\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline(self):
        """Test der kompletten Pipeline von Dataset bis DataLoader."""
        # Erstelle DataLoader
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=2,
            num_workers=0,
            target_size=(128, 128)
        )

        # Teste alle Batches
        total_samples = 0
        for batch_idx, (images, labels, boxes) in enumerate(dataloader):
            batch_size = images.shape[0]
            total_samples += batch_size

            # Prüfe Batch-Konsistenz
            self.assertEqual(images.shape[1:], (3, 128, 128))
            self.assertEqual(labels.shape[0], batch_size)
            self.assertEqual(len(boxes), batch_size)

            # Prüfe dass Labels im erwarteten Bereich sind
            self.assertTrue(torch.all(labels >= 0))
            self.assertTrue(torch.all(labels < 5))

            # Prüfe Bounding Box Konsistenz
            for i, box_tensor in enumerate(boxes):
                self.assertEqual(box_tensor.shape[1], 4)
                # Anzahl Boxen sollte dem erwarteten Muster entsprechen
                expected_boxes = (batch_idx * 2 + i) + 1
                if total_samples <= 5:  # Nur für vollständige Samples
                    self.assertEqual(box_tensor.shape[0], expected_boxes)

        # Prüfe dass alle Samples verarbeitet wurden
        self.assertEqual(total_samples, 5)


if __name__ == '__main__':
    # Führe alle Tests aus
    unittest.main(verbosity=2)


# Zusätzliche Benchmark-Tests (optional)
class BenchmarkTests:
    """Performance-Tests für das Dataset-Modul."""

    @staticmethod
    def benchmark_loading_speed(dataset_path: str, num_samples: int = 100):
        """Benchmark für Ladegeschwindigkeit."""
        import time

        # Hier würde ein echter Benchmark mit echten Daten laufen
        print(f"Benchmark für {num_samples} Samples...")

        start_time = time.time()
        # Simuliere Laden von Samples
        for i in range(num_samples):
            # dataset[i] würde hier aufgerufen
            pass
        end_time = time.time()

        avg_time = (end_time - start_time) / num_samples
        print(f"Durchschnittliche Ladezeit pro Sample: {avg_time * 1000:.2f} ms")
        print(f"Samples pro Sekunde: {1 / avg_time:.1f}")

    @staticmethod
    def benchmark_batch_processing(dataloader, num_batches: int = 10):
        """Benchmark für Batch-Verarbeitung."""
        import time

        times = []
        for i, (images, labels, boxes) in enumerate(dataloader):
            if i >= num_batches:
                break

            start_time = time.time()

            # Simuliere GPU-Transfer
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Simuliere Forward Pass
            _ = images.sum()

            end_time = time.time()
            times.append(end_time - start_time)

        if times:
            avg_time = sum(times) / len(times)
            print(f"Durchschnittliche Batch-Verarbeitungszeit: {avg_time * 1000:.2f} ms")


# Hilfsfunktionen für Tests
def create_mock_imagenet_structure(base_path: str, num_images: int = 10):
    """Erstellt eine Mock-ImageNet-Verzeichnisstruktur für Tests."""
    base_path = Path(base_path)
    image_dir = base_path / "images"
    annot_dir = base_path / "annotations"

    image_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)

    # Erstelle Mock-Bilder und Annotationen
    for i in range(num_images):
        # Bild
        img = Image.new('RGB', (224, 224), color=(i * 25, i * 20, i * 15))
        img.save(image_dir / f"ILSVRC2012_val_{i:08d}.JPEG")

        # XML-Annotation
        root = ET.Element("annotation")
        obj = ET.SubElement(root, "object")
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(i * 10)
        ET.SubElement(bbox, "ymin").text = str(i * 10)
        ET.SubElement(bbox, "xmax").text = str(i * 10 + 50)
        ET.SubElement(bbox, "ymax").text = str(i * 10 + 50)

        tree = ET.ElementTree(root)
        tree.write(annot_dir / f"ILSVRC2012_val_{i:08d}.xml")

    # Labels
    label_file = base_path / "labels.txt"
    with open(label_file, 'w') as f:
        for i in range(num_images):
            f.write(f"{(i % 1000) + 1}\n")

    return str(image_dir), str(annot_dir), str(label_file)


def run_all_tests():
    """Führt alle Tests aus und gibt einen Report aus."""
    print("🧪 Starte ImageNet Dataset Tests...")
    print("=" * 50)

    # Führe Unit Tests aus
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Zusammenfassung
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


# Beispiel für Test-Ausführung
if __name__ == "__main__":
    # Standardmäßig alle Tests ausführen
    success = run_all_tests()

    # Optional: Benchmarks (nur wenn echte Daten verfügbar)
    # benchmark = BenchmarkTests()
    # benchmark.benchmark_loading_speed("/path/to/data")