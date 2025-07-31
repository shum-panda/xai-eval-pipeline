import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Dict, Union

from pipeline_moduls.xai_methods.impl.score_cam.score_cam_config import ScoreCAMConfig
from src.pipeline_moduls.models.base.interface.xai_model import XAIModel
from src.pipeline_moduls.xai_methods.base.base_explainer import BaseExplainer
from src.pipeline_moduls.xai_methods.base.base_xai_config import BaseXAIConfig
from utils.with_cuda_cleanup import with_cuda_cleanup


class ScoreCamExplainer(BaseExplainer):
    """
    Explainer using a simplified ScoreCAM implementation.
    """

    def __init__(self, model: XAIModel, use_defaults: bool = True, **kwargs: object) -> None:
        self.target_layer = None
        self.feature_map = None
        self.hook_handle = None
        super().__init__(model, use_defaults, **kwargs)

    @with_cuda_cleanup
    def _compute_attributions(self, images: Tensor, target_classes: Tensor) -> Tensor:
        self._register_hook()

        with torch.no_grad():
            _ = self._model.pytorch_model(images)

        fmap = self.feature_map  # (B, C, H, W)
        B, C, H, W = fmap.shape
        attribution = torch.zeros((B, H, W), device=images.device)

        for i in range(C):
            upsampled = F.interpolate(fmap[:, i:i+1], size=images.shape[-2:], mode="bilinear", align_corners=False)
            masked_input = images * upsampled
            scores = self._model.get_predictions(masked_input)
            probs = torch.softmax(scores, dim=1)
            weights = probs[range(B), target_classes]
            attribution += weights.view(B, 1, 1) * upsampled.squeeze(1)

        self._remove_hook()
        attribution = F.relu(attribution)
        attribution = attribution / (attribution.max(dim=-1)[0].max(dim=-1)[0].view(B, 1, 1) + 1e-8)
        return attribution.detach().cpu()

    def _register_hook(self) -> None:
        layer = self._model.get_layer_by_name(self.target_layer)

        def forward_hook(module, input, output):
            self.feature_map = output.detach()

        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.hook_handle = layer.register_forward_hook(forward_hook)

    def _remove_hook(self) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _setup_with_validated_params(self, config: ScoreCAMConfig) -> None:
        self.target_layer = config.target_layer

    def check_input(self, **kwargs: Any) -> BaseXAIConfig:
        try:
            config = ScoreCAMConfig(**kwargs)
            config.validate()
        except (TypeError, ValueError) as e:
            self._logger.error(f"Invalid ScoreCAM config: {e}")
            raise

        config.target_layer = self._select_target_layer(config.target_layer)
        return config

    def _select_target_layer(self, layer_idx: Union[str, int]) -> str:
        """
        Select a convolutional layer as the ScoreCAM target.

        Args:
            layer_idx (Union[str, int]): Index or name of the desired layer.

        Returns:
            str: name of The selected target layer.
        #todo: abstract cam
        """
        model = self._model.pytorch_model
        conv_modules: list[tuple[str, nn.Module]] = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_modules.append((name, module))

        if not conv_modules:
            self._logger.warning("No Conv2d layers found. Returning the _model itself.")
            raise ValueError("this GradCam expected Convolutional Layer ")

        if isinstance(layer_idx, str):
            for name, module in conv_modules:
                if name == layer_idx:
                    selected = (name, module)
                    break
            else:
                self._logger.warning(
                    f"Layer name {layer_idx} not found. Using last conv layer."
                )
                selected = conv_modules[-1]
        elif isinstance(layer_idx, int):
            if layer_idx == -1 or layer_idx >= len(conv_modules):
                self._logger.info("Using last convolutional layer.")
                selected = conv_modules[-1]
            elif 0 <= layer_idx < len(conv_modules):
                selected = conv_modules[layer_idx]
            else:
                self._logger.warning(
                    f"Layer index {layer_idx} out of range. Using last layer."
                )
                selected = conv_modules[-1]
        else:
            raise TypeError("layer_idx must be a string or integer.")

        return selected[0]

    @property
    def parameters(self) -> Dict[str, str]:
        return {"target_layer": str(self.target_layer)}

    @classmethod
    def get_name(cls) -> str:
        return "score_cam"
