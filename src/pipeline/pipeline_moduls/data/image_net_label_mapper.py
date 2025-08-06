import json
from pathlib import Path
from typing import Dict, Optional

import torch


class ImageNetLabelMapper:
    """
    Handles mappings between model class indices, ImageNet val indices,
    WordNet IDs, and human-readable labels.
    """

    def __init__(
        self,
        mapping_file: Optional[Path] = None,
        imagenet_class_index_file: Optional[Path] = None,
    ):
        if mapping_file is None:
            project_root = Path(__file__).resolve().parents[4]
            mapping_file = project_root / "data" / "raw" / "final_mapping.txt"

        if imagenet_class_index_file is None:
            raise ValueError("Please provide path to imagenet_class_index.json")

        self.val_to_wnid: Dict[int, str] = {}
        self.wnid_to_class: Dict[str, int] = {}
        self.class_to_val: Dict[int, int] = {}

        self._load_mapping(mapping_file)
        self.class_to_val_tensor = self._make_class_to_val_tensor()

        # Lade JSON-Mapping und baue Lookup-Tabellen
        with open(imagenet_class_index_file, "r") as f:
            class_idx_data = json.load(f)

        # wnid → label (string)
        self.wnid_to_label_map: Dict[str, str] = {}
        # class_idx → label (string)
        self.class_id_to_label: Dict[int, str] = {}

        for idx_str, (wnid, label) in class_idx_data.items():
            idx = int(idx_str)
            self.wnid_to_label_map[wnid] = label
            self.class_id_to_label[idx] = label

    def _load_mapping(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            val_idx_str, wnid, class_idx_str = line.strip().split()
            val_idx = int(val_idx_str)
            class_idx = int(class_idx_str)

            self.val_to_wnid[val_idx] = wnid
            self.wnid_to_class[wnid] = class_idx
            self.class_to_val[class_idx] = val_idx

    def _make_class_to_val_tensor(self) -> torch.Tensor:
        max_class_idx = max(self.class_to_val.keys())
        tensor = torch.full((max_class_idx + 1,), -1, dtype=torch.long)
        for class_idx, val_idx in self.class_to_val.items():
            tensor[class_idx] = val_idx
        return tensor

    def class_to_val(self, class_idx: int) -> int:
        return self.class_to_val[class_idx]

    def val_to_class(self, val_idx: int) -> int:
        wnid = self.val_to_wnid[val_idx]
        return self.wnid_to_class[wnid]

    def wnid_to_label(self, wnid: str) -> str:
        return self.wnid_to_label_map.get(wnid, f"[invalid wnid: {wnid}]")

    def class_to_label(self, class_idx: int) -> str:
        return self.class_id_to_label.get(class_idx, "[unknown]")

    def val_to_label(self, val_idx: int) -> str:
        wnid = self.val_to_wnid.get(val_idx)
        if wnid is None:
            return f"[invalid val idx: {val_idx}]"
        return self.wnid_to_label(wnid)

    def map_class_tensor_to_val(self, class_indices: torch.Tensor) -> torch.Tensor:
        return self.class_to_val_tensor[class_indices.cpu()]
