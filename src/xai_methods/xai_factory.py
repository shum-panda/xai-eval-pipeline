from xai_methods.MemoryManagement.adaptive_batch_processor import AdaptiveBatchProcessor
from xai_methods.MemoryManagement import DirectBatchProcessor
from xai_methods.implementations.grand_cam_explainer import GradCamExplainer


class XAIFactory:
    """Factory ohne zirkuläre Abhängigkeiten."""

    @staticmethod
    def create_gradcam_explainer(model,
                                 layer: int = -1,
                                 enable_adaptive_batching: bool = True,
                                 **memory_kwargs) -> GradCamExplainer:
        """
        Erstellt GradCAM-Explainer mit gewählter Memory-Strategy.
        """
        # Strategy basierend auf Parameter wählen
        if enable_adaptive_batching:
            batch_processor = AdaptiveBatchProcessor(**memory_kwargs)
        else:
            batch_processor = DirectBatchProcessor()

        # Explainer mit injizierter Strategy erstellen
        return GradCamExplainer(
            model=model,
            layer=layer,
            batch_processor=batch_processor
        )