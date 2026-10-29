import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image

from src.pipe.moduls.data.image_net_val_dataset import ImageNetValDataset


class TestBoxScalingNonSquare(unittest.TestCase):
    """Boxes must be scaled per axis: x with the width, y with the height."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        image_dir = self.temp_dir / "images"
        annot_dir = self.temp_dir / "annotations"
        image_dir.mkdir()
        annot_dir.mkdir()

        # Landscape image like most of ImageNet: 400 wide, 200 high
        Image.new("RGB", (400, 200)).save(image_dir / "ILSVRC2012_val_00000001.JPEG")
        root = ET.Element("annotation")
        bndbox = ET.SubElement(ET.SubElement(root, "object"), "bndbox")
        for key, value in {"xmin": 200, "ymin": 50, "xmax": 400, "ymax": 150}.items():
            ET.SubElement(bndbox, key).text = str(value)
        ET.ElementTree(root).write(annot_dir / "ILSVRC2012_val_00000001.xml")

        label_file = self.temp_dir / "labels.txt"
        label_file.write_text("1\n")

        self.dataset = ImageNetValDataset(
            image_dir=image_dir,
            annot_dir=annot_dir,
            label_file=label_file,
            transform=None,
            target_size=[224, 224],
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_landscape_box_is_scaled_per_axis(self) -> None:
        xml_path = self.temp_dir / "annotations" / "ILSVRC2012_val_00000001.xml"
        boxes = self.dataset._get_boxes(xml_path, (400, 200))  # PIL size is (W, H)

        # x: 224 / 400 = 0.56, y: 224 / 200 = 1.12
        expected = torch.tensor([[112.0, 56.0, 224.0, 168.0]])
        torch.testing.assert_close(boxes, expected)


if __name__ == "__main__":
    unittest.main()
