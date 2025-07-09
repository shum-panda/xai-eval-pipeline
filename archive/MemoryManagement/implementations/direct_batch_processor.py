import torch

from archive.MemoryManagement.base.batch_processor import BatchProcessor
from archive.MemoryManagement.base.explainer_callable import ExplainerCallable


class DirectBatchProcessor(BatchProcessor):
    """Direkte Verarbeitung ohne Memory Management."""

    def process_batch(
        self, images: torch.Tensor, explain_fn: ExplainerCallable
    ) -> torch.Tensor:
        return explain_fn(images)
