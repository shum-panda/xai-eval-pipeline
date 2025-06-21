import gc
import logging

import torch

from archive.MemoryManagement.base.batch_processor import BatchProcessor
from archive.MemoryManagement.base.explainer_callable import ExplainerCallable


class AdaptiveBatchProcessor(BatchProcessor):
    """Adaptive Batch-Verarbeitung mit Memory Management."""

    def __init__(self,
                 target_memory_usage: float = 0.8,
                 kp: float = 0.1,
                 max_batch_size: int = 32,
                 min_batch_size: int = 1):
        self.target_memory_usage = target_memory_usage
        self.kp = kp
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.current_batch_size = min_batch_size

        self.logger = logging.getLogger(__name__)
        self.memory_history = []
        self.max_history = 5

    def process_batch(self,
                      images: torch.Tensor,
                      explain_fn: ExplainerCallable) -> torch.Tensor:
        """
        Verarbeitet Bilder mit adaptivem Batching.

        KEIN Zirkulärer Dependency - explain_fn ist nur ein Callback!
        """
        if len(images) <= self.min_batch_size:
            return explain_fn(images)

        # Initial batch size bestimmen
        if self.current_batch_size == self.min_batch_size:
            self.current_batch_size = self._estimate_initial_batch_size(images[0:1], explain_fn)

        all_results = []
        processed_count = 0
        total_images = len(images)

        self.logger.info(f"Processing {total_images} images with adaptive batching")

        while processed_count < total_images:
            end_idx = min(processed_count + self.current_batch_size, total_images)
            current_batch = images[processed_count:end_idx]

            try:
                # Memory vor Verarbeitung
                memory_before = self._get_gpu_memory_usage()

                # HIER: Callback aufrufen - kein Objekt-Referenz!
                batch_results = explain_fn(current_batch)

                # Memory nach Verarbeitung
                memory_after = self._get_gpu_memory_usage()

                all_results.append(batch_results)
                processed_count += len(current_batch)

                # Batch-Size anpassen
                self._adjust_batch_size(memory_after)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"OOM: reducing batch size from {self.current_batch_size}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.current_batch_size = max(self.min_batch_size,
                                                  self.current_batch_size // 2)
                    continue
                else:
                    raise e

        # Ergebnisse concatenieren
        return torch.cat(all_results, dim=0)

    def _estimate_initial_batch_size(self,
                                     sample: torch.Tensor,
                                     explain_fn: ExplainerCallable) -> int:
        """Schätzt sichere Initial-Batch-Size."""
        try:
            # Test mit Sample
            memory_before = self._get_gpu_memory_usage()
            _ = explain_fn(sample)
            memory_after = self._get_gpu_memory_usage()

            memory_per_sample = memory_after - memory_before
            if memory_per_sample <= 0:
                return self.max_batch_size // 4  # Conservative fallback

            available_memory = (1.0 - memory_before) * self.target_memory_usage
            estimated_batch_size = int(available_memory / memory_per_sample)

            return max(self.min_batch_size,
                       min(estimated_batch_size, self.max_batch_size))

        except Exception as e:
            self.logger.warning(f"Error estimating batch size: {e}")
            return self.min_batch_size

    def _adjust_batch_size(self, memory_usage: float) -> None:
        """Passt Batch-Size basierend auf Memory-Usage an."""
        self.memory_history.append(memory_usage)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)

        target_diff = self.target_memory_usage - memory_usage
        threshold = 0.05

        if abs(target_diff) > threshold:
            adjustment_factor = 1.0 + (self.kp * target_diff)
            new_batch_size = int(self.current_batch_size * adjustment_factor)
            new_batch_size = max(self.min_batch_size,
                                 min(new_batch_size, self.max_batch_size))

            if new_batch_size != self.current_batch_size:
                self.logger.debug(f"Batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size

    def _get_gpu_memory_usage(self) -> float:
        """GPU Memory Usage als Ratio."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        return 0.5