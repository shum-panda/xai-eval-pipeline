import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List

import torch
from PIL import Image

from src.pipe.moduls.data.dataclass.image_net_sample import ImageNetSample
from src.pipe.moduls.data.dataclass.xai_input_batch import XAIInputBatch
from src.pipe.moduls.data.image_net_val_dataset import (
    ImageNetValDataset,
    create_dataloader,
)
from src.pipe.moduls.data.utils.bbox_to_mask import parse_bbox
from src.pipe.moduls.data.utils.collate_fn import explain_collate_fn


def write_annotation(xml_path: Path, boxes: List[List[int]]) -> None:
    root = ET.Element("annotation")
    for box in boxes:
        bndbox = ET.SubElement(ET.SubElement(root, "object"), "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bndbox, key).text = str(value)
    ET.ElementTree(root).write(xml_path)


class ImageNetFixture(unittest.TestCase):
    """Three 100x80 images in ImageNet naming with 2, 1 and 0 boxes."""

    BOXES = [[[10, 10, 50, 50], [60, 20, 90, 70]], [[20, 15, 80, 65]], []]

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.image_dir = self.temp_dir / "images"
        self.annot_dir = self.temp_dir / "annotations"
        self.label_file = self.temp_dir / "labels.txt"
        self.image_dir.mkdir()
        self.annot_dir.mkdir()

        for i, boxes in enumerate(self.BOXES):
            name = f"ILSVRC2012_val_{i + 1:08d}"
            Image.new("RGB", (100, 80), color=(i * 80, 0, 0)).save(
                self.image_dir / f"{name}.JPEG"
            )
            write_annotation(self.annot_dir / f"{name}.xml", boxes)
        self.label_file.write_text("490\n361\n171\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def make_dataset(self, target_size: Any = (224, 224)) -> ImageNetValDataset:
        return ImageNetValDataset(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            transform=None,
            target_size=target_size,
        )


class TestImageNetValDataset(ImageNetFixture):

    def test_labels_are_read_in_file_order(self) -> None:
        dataset = self.make_dataset()
        self.assertEqual(len(dataset), 3)
        # Devkit IDs as in the ground truth file; mapped to PyTorch indices later
        self.assertEqual(dataset.labels, [490, 361, 171])

    def test_getitem_returns_resized_sample(self) -> None:
        sample = self.make_dataset()[0]

        self.assertIsInstance(sample, ImageNetSample)
        self.assertEqual(sample.image_tensor.shape, (3, 224, 224))
        self.assertEqual(sample.label, 490)
        self.assertEqual(sample.label_tensor.dtype, torch.long)
        self.assertEqual(sample.bbox_tensor.shape, (2, 4))
        self.assertEqual(sample.image_name, "ILSVRC2012_val_00000001.JPEG")

    def test_boxes_are_scaled_to_target_size(self) -> None:
        sample = self.make_dataset()[1]
        # 100x80 -> 224x224: x * 2.24, y * 2.8
        expected = torch.tensor([[44.8, 42.0, 179.2, 182.0]])
        torch.testing.assert_close(sample.bbox_tensor, expected)

    def test_image_without_boxes_has_empty_box_tensor(self) -> None:
        sample = self.make_dataset()[2]
        self.assertEqual(sample.bbox_tensor.shape, (0, 4))

    def test_missing_annotation_gives_empty_boxes(self) -> None:
        (self.annot_dir / "ILSVRC2012_val_00000002.xml").unlink()
        sample = self.make_dataset()[1]
        self.assertEqual(sample.bbox_tensor.shape, (0, 4))

    def test_get_sample_info(self) -> None:
        info = self.make_dataset().get_sample_info(0)
        self.assertEqual(info["label"], 490)
        self.assertTrue(info["exists_xml"])


class TestParseBbox(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_parse_valid_xml(self) -> None:
        xml_path = self.temp_dir / "test.xml"
        write_annotation(xml_path, [[10, 20, 30, 40], [50, 60, 70, 80]])
        expected = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32
        )
        torch.testing.assert_close(parse_bbox(xml_path), expected)

    def test_parse_xml_without_objects(self) -> None:
        xml_path = self.temp_dir / "empty.xml"
        write_annotation(xml_path, [])
        self.assertEqual(parse_bbox(xml_path).shape, (0, 4))

    def test_missing_xml_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            parse_bbox(self.temp_dir / "nonexistent.xml")


class TestCollateAndDataLoader(ImageNetFixture):

    def test_collate_builds_xai_input_batch(self) -> None:
        dataset = self.make_dataset()
        batch = explain_collate_fn([dataset[0], dataset[2]])

        self.assertIsInstance(batch, XAIInputBatch)
        self.assertEqual(batch.images_tensor.shape, (2, 3, 224, 224))
        self.assertEqual(batch.labels_tensor.tolist(), [490, 171])
        self.assertEqual([b.shape[0] for b in batch.boxes_list], [2, 0])
        self.assertEqual(batch.labels_int, [490, 171])

    def test_create_dataloader_batches_all_samples(self) -> None:
        dataloader = create_dataloader(
            image_dir=self.image_dir,
            annot_dir=self.annot_dir,
            label_file=self.label_file,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            target_size=[128, 128],
            transform=None,
        )
        batches = list(dataloader)

        self.assertEqual([len(b.labels_int) for b in batches], [2, 1])
        self.assertEqual(batches[0].images_tensor.shape[1:], (3, 128, 128))


if __name__ == "__main__":
    unittest.main()
