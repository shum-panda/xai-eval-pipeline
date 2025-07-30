import logging
from pathlib import Path
from typing import Optional, Union

import torch


class AttributionReference:
    """
    Reference to an attribution file on disk instead of keeping tensor in memory.
    Provides lazy loading when attribution is actually needed.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._cached_attribution: Optional[torch.Tensor] = None
        self._logger = logging.getLogger(__name__)

    @property
    def attribution(self) -> torch.Tensor:
        """
        Lazy load attribution tensor from disk when accessed.
        Caches it for subsequent access within the same operation.
        """
        if self._cached_attribution is None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Attribution file not found: {self.file_path}")

            self._cached_attribution = torch.load(self.file_path, map_location="cpu")
            self._logger.debug(f"Loaded attribution from {self.file_path}")

        return self._cached_attribution

    def clear_cache(self):
        """Clear cached attribution to free memory"""
        if self._cached_attribution is not None:
            del self._cached_attribution
            self._cached_attribution = None
            import gc

            gc.collect()

    @property
    def shape(self) -> tuple:
        """Get shape without loading full tensor (if possible)"""
        # For now, load to get shape - could be optimized with metadata
        return self.attribution.shape

    def __str__(self):
        return f"AttributionReference({self.file_path})"

    def __repr__(self):
        return (
            f"AttributionReference(file_path='{self.file_path}', cached"
            f"={self._cached_attribution is not None})"
        )
