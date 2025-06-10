import torch

from xai_methods.MemoryManagement.base.batch_processor import BatchProcessor
from xai_methods.MemoryManagement.base.explainer_callable import ExplainerCallable


class DirectBatchProcessor(BatchProcessor):
    """Direkte Verarbeitung ohne Memory Management."""

    def process_batch(self,
                      images: torch.Tensor,
                      explain_fn: ExplainerCallable) -> torch.Tensor:
        return explain_fn(images)