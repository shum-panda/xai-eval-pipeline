import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.pipeline.pipeline_moduls.data.dataclass.image_net_sample import ImageNetSample
from src.pipeline.pipeline_moduls.data.dataclass.xai_input_batch import XAIInputBatch
from src.pipeline.pipeline_moduls.data.image_net_val_dataset import (
    ImageNetValDataset,
    create_dataloader,
)
from src.pipeline.pipeline_moduls.data.utils.bbox_to_mask import parse_bbox
from src.pipeline.pipeline_moduls.data.utils.collate_fn import explain_collate_fn
from src.pipeline.pipeline_moduls.utils.bbox_to_mask_tensor import bbox_to_mask_tensor


class TestImageNetValDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"

        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        self._create_test_data()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_test_data(self):
        # Create test images
        self.test_images = []
        for i in range(3):
            img = Image.new("RGB", (100, 80), color=(i * 80, i * 60, i * 40))
            img_path = self.image_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
            img.save(img_path)
            self.test_images.append(img_path)

        # Create test annotations
        test_boxes = [
            [[10, 10, 50, 50], [60, 20, 90, 70]],  # Image 0: 2 boxes
            [[20, 15, 80, 65]],  # Image 1: 1 box
            [],  # Image 2: 0 boxes
        ]

        for i, boxes in enumerate(test_boxes):
            xml_path = self.annot_dir / f"ILSVRC2012_val_{i:08d}.xml"
            self._create_xml_annotation(xml_path, boxes, (100, 80))

        # Create labels file
        with open(self.label_file, "w") as f:
            f.write("1\n2\n3\n")

    def _create_xml_annotation(self, xml_path: Path, boxes: list, image_size: tuple):
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

    def test_dataset_initialization(self):
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
        )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.labels), 3)
        self.assertEqual(dataset.labels, [0, 1, 2])  # Zero-indexed

    def test_dataset_getitem(self):
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
        self.assertEqual(boxes.shape, (2, 4))  # 2 boxes, 4 coordinates each

    def test_dataset_missing_annotation(self):
        # Remove one annotation file
        annotation_to_remove = self.annot_dir / "ILSVRC2012_val_00000001.xml"
        annotation_to_remove.unlink()

        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
        )

        image, label, boxes = dataset[1]
        self.assertEqual(len(boxes), 0)  # No boxes when annotation is missing

    def test_get_sample_info(self):
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

    def test_dataset_different_target_sizes(self):
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            target_size=(128, 128),
        )

        image, _, _ = dataset[0]
        self.assertEqual(image.shape, (3, 128, 128))


class TestBboxToMask(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_parse_valid_xml(self):
        xml_path = Path(self.temp_dir) / "test.xml"

        root = ET.Element("annotation")
        for box in [[10, 20, 30, 40], [50, 60, 70, 80]]:
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

    def test_parse_nonexistent_xml(self):
        xml_path = Path(self.temp_dir) / "nonexistent.xml"
        boxes = parse_bbox(xml_path)

        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)

    def test_parse_empty_xml(self):
        xml_path = Path(self.temp_dir) / "empty.xml"

        root = ET.Element("annotation")
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        boxes = parse_bbox(xml_path)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)


class TestCollateFn(unittest.TestCase):

    def test_collate_fn_basic(self):
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

    def test_collate_fn_empty_boxes(self):
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0), torch.empty(0, 4)),
            (torch.randn(3, 224, 224), torch.tensor(1), torch.randn(2, 4)),
        ]

        images, labels, boxes = explain_collate_fn(batch)

        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(boxes[0]), 0)
        self.assertEqual(boxes[1].shape, (2, 4))

    def test_collate_fn_single_item(self):
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(5), torch.randn(1, 4)),
        ]

        images, labels, boxes = explain_collate_fn(batch)

        self.assertEqual(images.shape, (1, 3, 224, 224))
        self.assertEqual(labels.shape, (1,))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].shape, (1, 4))


class TestCreateDataLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"

        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        # Create minimal test data
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

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_dataloader_basic(self):
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

    def test_create_dataloader_custom_params(self):
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=2,
            num_workers=0,
            target_size=(128, 128),
            shuffle=False,
        )

        self.assertIsInstance(dataloader, DataLoader)

        for images, labels, boxes in dataloader:
            self.assertEqual(images.shape[1:], (3, 128, 128))
            break


class TestImageNetSample(unittest.TestCase):

    def test_sample_creation(self):
        sample = ImageNetSample(
            image_path="/path/to/image.jpg",
            image_name="image.jpg",
            label=5,
            bbox_coordinates=torch.tensor([[10, 20, 30, 40]]),
        )

        self.assertEqual(sample.image_path, "/path/to/image.jpg")
        self.assertEqual(sample.image_name, "image.jpg")
        self.assertEqual(sample.label, 5)
        self.assertEqual(sample.bbox_coordinates.shape, (1, 4))

    def test_sample_with_empty_bbox(self):
        sample = ImageNetSample(
            image_path="/path/to/image.jpg",
            image_name="image.jpg",
            label=5,
            bbox_coordinates=torch.empty(0, 4),
        )

        self.assertEqual(len(sample.bbox_coordinates), 0)


class TestXAIInputBatch(unittest.TestCase):

    def test_batch_creation(self):
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([1, 5])
        bbox_list = [torch.tensor([[10, 20, 30, 40]]), torch.empty(0, 4)]

        batch = XAIInputBatch(images=images, labels=labels, bbox_list=bbox_list)

        self.assertEqual(batch.images.shape, (2, 3, 224, 224))
        self.assertEqual(batch.labels.shape, (2,))
        self.assertEqual(len(batch.bbox_list), 2)
        self.assertEqual(batch.bbox_list[0].shape, (1, 4))
        self.assertEqual(len(batch.bbox_list[1]), 0)


class TestBboxToMaskTensor(unittest.TestCase):

    def test_bbox_to_mask_tensor_single_box(self):
        bbox = torch.tensor([[10, 20, 30, 40]])  # xmin, ymin, xmax, ymax
        mask = bbox_to_mask_tensor(bbox, height=50, width=60)

        self.assertEqual(mask.shape, (50, 60))

        # Check that the mask is 1 inside the box and 0 outside
        self.assertEqual(mask[25, 20].item(), 1.0)  # Inside box
        self.assertEqual(mask[25, 35].item(), 1.0)  # Inside box
        self.assertEqual(mask[10, 20].item(), 0.0)  # Outside box (above)
        self.assertEqual(mask[25, 5].item(), 0.0)  # Outside box (left)

    def test_bbox_to_mask_tensor_multiple_boxes(self):
        bbox = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
        mask = bbox_to_mask_tensor(bbox, height=50, width=50)

        self.assertEqual(mask.shape, (50, 50))

        # Check that both boxes are marked in the mask
        self.assertEqual(mask[15, 15].item(), 1.0)  # First box
        self.assertEqual(mask[35, 35].item(), 1.0)  # Second box
        self.assertEqual(mask[25, 25].item(), 0.0)  # Between boxes

    def test_bbox_to_mask_tensor_empty_bbox(self):
        bbox = torch.empty(0, 4)
        mask = bbox_to_mask_tensor(bbox, height=50, width=50)

        self.assertEqual(mask.shape, (50, 50))
        self.assertEqual(mask.sum().item(), 0.0)  # All zeros

    def test_bbox_to_mask_tensor_out_of_bounds(self):
        # Bbox partially outside image bounds
        bbox = torch.tensor([[-5, -5, 10, 10]])
        mask = bbox_to_mask_tensor(bbox, height=20, width=20)

        self.assertEqual(mask.shape, (20, 20))
        # Should only mark the part that's inside the image
        self.assertEqual(mask[5, 5].item(), 1.0)  # Inside both box and image

    def test_bbox_to_mask_tensor_dtype_and_device(self):
        bbox = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
        mask = bbox_to_mask_tensor(bbox, height=30, width=30)

        self.assertEqual(mask.dtype, torch.float32)
        if torch.cuda.is_available():
            bbox_cuda = bbox.cuda()
            mask_cuda = bbox_to_mask_tensor(bbox_cuda, height=30, width=30)
            self.assertTrue(mask_cuda.is_cuda)


class TestDataProcessingIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.annot_dir = Path(self.temp_dir) / "annotations"
        self.label_file = Path(self.temp_dir) / "labels.txt"

        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        # Create multiple test images with varying bounding boxes
        for i in range(5):
            img = Image.new("RGB", (200, 150), color=(i * 50, i * 40, i * 30))
            img.save(self.image_dir / f"img_{i:03d}.JPEG")

            root = ET.Element("annotation")
            for j in range(i + 1):  # Variable number of boxes per image
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

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_full_data_pipeline(self):
        # Create dataset
        dataset = ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            target_size=(128, 128),
        )

        # Create dataloader
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=2,
            num_workers=0,
            target_size=(128, 128),
        )

        # Test one batch
        for images, labels, boxes in dataloader:
            batch_size = images.shape[0]

            # Verify image shapes
            self.assertEqual(images.shape[1:], (3, 128, 128))
            self.assertEqual(labels.shape[0], batch_size)
            self.assertEqual(len(boxes), batch_size)

            # Test conversion to masks
            for i, bbox_tensor in enumerate(boxes):
                if len(bbox_tensor) > 0:
                    mask = bbox_to_mask_tensor(bbox_tensor, height=128, width=128)
                    self.assertEqual(mask.shape, (128, 128))
                    self.assertGreater(
                        mask.sum().item(), 0.0
                    )  # Should have some positive pixels

            break  # Test only first batch


if __name__ == "__main__":
    unittest.main()
