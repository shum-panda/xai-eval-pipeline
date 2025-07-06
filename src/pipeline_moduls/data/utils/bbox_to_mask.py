import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional


import numpy as np
from sympy.printing.pytorch import torch


def bbox_to_mask(bbox_info: Dict, image_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    """
    Konvertiere Bounding Box zu Binary Mask für XAI Evaluation.

    Args:
        bbox_info: Bounding Box Info vom Dataset
        image_size: (width, height) des Bildes

    Returns:
        Binary mask tensor
    """
    if not bbox_info or not bbox_info['objects']:
        return None

    width, height = image_size
    mask = torch.zeros((height, width), dtype=torch.float32)

    for obj in bbox_info['objects']:
        if 'bbox' not in obj:
            continue #Robustness against faulty or incomplete data
        x1, y1, x2, y2 = obj['bbox']
        # Round & clamp Koordinaten
        x1 = max(0, min(width, int(round(x1))))
        x2 = max(0, min(width, int(round(x2))))
        y1 = max(0, min(height, int(round(y1))))
        y2 = max(0, min(height, int(round(y2))))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    return mask

def parse_bbox(xml_path: Path) -> torch.Tensor:
    """
    Parst Bounding Boxes aus Pascal VOC XML-Datei und gibt sie als Tensor zurück.

    Args:
        xml_path: Pfad zur XML-Datei

    Returns:
        torch.Tensor: Bounding Boxes im Format [N, 4] mit [xmin, ymin, xmax, ymax]

    Raises:
        FileNotFoundError: Wenn XML-Datei nicht existiert
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])

        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    except ET.ParseError:
        warnings.warn(f"Fehler beim Parsen von {xml_path}")
        return torch.empty((0, 4), dtype=torch.float32)

