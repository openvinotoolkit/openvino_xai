# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copy & edit from https://github.com/openvinotoolkit/training_extensions/blob/2.1.0/src/otx/algo/explain/explain_algo.py
"""Algorithms for calculcalating XAI branch for Explainable AI."""

import copy
from typing import Any, Callable, Dict, Mapping

import numpy as np
import torch

from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME, has_xai
from openvino_xai.methods.base import IdentityPreprocessFN, MethodBase


class TorchWhiteBoxMethod(MethodBase[torch.nn.Module, torch.nn.Module]):
    """
    Base class for Torch-based methods.

    :param model: Input model.
    :type model: torch.nn.Module
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :type target_layer: str
    :param embed_scaling: Whether to scale output or not.
    :type embed_scaling: bool
    :param device_name: Device type name.
    :type device_name: str
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        target_layer: str | None = None,
        embed_scaling: bool = True,
        device_name: str = "CPU",
        **kwargs,
    ):
        super().__init__(model=model, preprocess_fn=preprocess_fn, device_name=device_name)
        self._target_layer = target_layer
        self._embed_scaling = embed_scaling

    def prepare_model(self, load_model: bool = True) -> torch.nn.Module:
        """Return XAI inserted model."""
        if has_xai(self._model):
            if load_model:
                self._model_compiled = self._model
            return self._model

        model = copy.deepcopy(self._model)
        # Feature
        feature_layer = model.get_submodule(self._target_layer)
        feature_layer.register_forward_hook(self._feature_hook)
        # Output
        model.register_forward_hook(self._output_hook)
        setattr(model, "has_xai", True)
        model.eval()

        if load_model:
            self._model_compiled = model
        return model

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> Mapping:
        """Process numpy input, return numpy output."""
        if not self._model_compiled:
            raise RuntimeError("Model is not compiled. Call prepare_model() first.")

        if preprocess:
            x = self.preprocess_fn(x)
        x = torch.from_numpy(x).float()

        with torch.no_grad():
            x = self._model_compiled(x)

        output = {}
        for name, data in x.items():
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            output[name] = data.numpy(force=True)
        return output

    def _feature_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        self._feature_map = output
        return output

    def _output_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "prediction": output,
            SALIENCY_MAP_OUTPUT_NAME: torch.empty_like(output),
        }

    def generate_saliency_map(self, data: np.ndarray) -> np.ndarray:
        """Return saliency map."""
        model_output = self.model_forward(data)
        return model_output[SALIENCY_MAP_OUTPUT_NAME]

    @staticmethod
    def _normalize_map(saliency_map: torch.Tensor) -> torch.Tensor:
        """Normalize saliency maps."""
        max_values = saliency_map.max(dim=-1, keepdim=True).values
        min_values = saliency_map.min(dim=-1, keepdim=True).values
        saliency_map = 255 * (saliency_map - min_values) / (max_values - min_values + 1e-12)
        return saliency_map.to(torch.uint8)


class TorchActivationMap(TorchWhiteBoxMethod):
    """ActivationMap. Mean of the feature map along the channel dimension."""

    def _output_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_map = self._feature_map
        batch_size, _, h, w = feature_map.shape
        activation_map = torch.mean(feature_map, dim=1)
        if self._embed_scaling:
            activation_map = activation_map.reshape((batch_size, h * w))
            activation_map = self._normalize_map(activation_map)
            activation_map = activation_map.reshape((batch_size, h, w))
        return {
            "prediction": output,
            SALIENCY_MAP_OUTPUT_NAME: activation_map,
        }


class TorchReciproCAM(TorchWhiteBoxMethod):
    """Implementation of Recipro-CAM for class-wise saliency map.

    Recipro-CAM: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)

    :param optimize_gap: Whether to optimize out Global Average Pooling operation
    :type optimizae_gap: bool
    """

    def __init__(self, *args, optimize_gap: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimize_gap = optimize_gap

    def _feature_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        """feature_maps -> vertical stack of feature_maps + mosaic_feature_maps."""
        batch_size, c, h, w = self._feature_shape = output.shape
        feature_map = output
        if self._optimize_gap:
            feature_map = feature_map.reshape([batch_size, c, h * w]).mean(dim=-1)[:, :, None, None]  # Spatial average
        feature_maps = [feature_map]
        for i in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(output[i], c, h, w)
            feature_maps.append(mosaic_feature_map)
        return torch.cat(feature_maps)

    def _output_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _, h, w = self._feature_shape
        num_classes = output.shape[1]
        predictions = output[:batch_size]
        saliency_maps = output[batch_size:]
        saliency_maps = saliency_maps.reshape([batch_size, h * w, num_classes])
        saliency_maps = saliency_maps.transpose(1, 2)  # BxHWxC -> BxCxHW
        if self._embed_scaling:
            saliency_maps = saliency_maps.reshape((batch_size * num_classes, h * w))
            saliency_maps = self._normalize_map(saliency_maps)
        saliency_maps = saliency_maps.reshape([batch_size, num_classes, h, w])
        return {
            "prediction": predictions,
            SALIENCY_MAP_OUTPUT_NAME: saliency_maps,
        }

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        if self._optimize_gap:
            # if isinstance(model_neck, GlobalAveragePooling):
            # Optimization workaround for the GAP case (simulate GAP with more simple compute graph)
            # Possible due to static sparsity of mosaic_feature_map
            # Makes the downstream GAP operation to be dummy
            feature_map_transposed = torch.flatten(feature_map, start_dim=1).transpose(0, 1)[:, :, None, None]
            mosaic_feature_map = feature_map_transposed / (h * w)
        else:
            feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
            mosaic_feature_map_mask = torch.zeros(h * w, c, h, w).to(feature_map.device)
            spatial_order = torch.arange(h * w).reshape(h, w)
            for i in range(h):
                for j in range(w):
                    k = spatial_order[i, j]
                    mosaic_feature_map_mask[k, :, i, j] = torch.ones(c).to(feature_map.device)
            mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask
        return mosaic_feature_map


class TorchViTReciproCAM(TorchReciproCAM):
    """Implementation of ViTRecipro-CAM for class-wise saliency map for transformer-based classifiers.

    ViT-ReciproCAM: Gradient and Attention-Free Visual Explanations for Vision Transformer
    (https://arxiv.org/abs/2310.02588)

    :param use_gaussian: Defines kernel type for mosaic feature map generation.
        If True, use gaussian 3x3 kernel. If False, use 1x1 kernel.
    :type use_gaussian: bool
    :param use_cls_token: If True, includes classification token into the mosaic feature map.
    :type use_cls_token: bool
    """

    def __init__(
        self,
        *args,
        use_gaussian: bool = True,
        use_cls_token: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._use_gaussian = use_gaussian
        self._use_cls_token = use_cls_token

    def _feature_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        """feature_maps -> vertical stack of feature_maps + mosaic_feature_maps."""
        feature_map = output
        batch_size, num_tokens, dim = feature_map.shape
        h = w = int((num_tokens - 1) ** 0.5)
        feature_maps = [feature_map]
        self._feature_shape = (batch_size, dim, h, w)
        for i in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[i], dim, h, w)
            feature_maps.append(mosaic_feature_map)
        return torch.cat(feature_maps)

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        num_tokens = h * w + 1
        mosaic_feature_map = torch.zeros(h * w, num_tokens, c).to(feature_map.device)

        if self._use_gaussian:
            if self._use_cls_token:
                mosaic_feature_map[:, 0, :] = feature_map[0, :]
            feature_map_spatial = feature_map[1:, :].reshape(1, h, w, c)
            feature_map_spatial_repeated = feature_map_spatial.repeat(h * w, 1, 1, 1)  # 196, 14, 14, 192

            spatial_order = torch.arange(h * w).reshape(h, w)
            gaussian = torch.tensor(
                [[1 / 16.0, 1 / 8.0, 1 / 16.0], [1 / 8.0, 1 / 4.0, 1 / 8.0], [1 / 16.0, 1 / 8.0, 1 / 16.0]],
            ).to(feature_map.device)
            mosaic_feature_map_mask_padded = torch.zeros(h * w, h + 2, w + 2).to(feature_map.device)
            for i in range(h):
                for j in range(w):
                    k = spatial_order[i, j]
                    i_pad = i + 1
                    j_pad = j + 1
                    mosaic_feature_map_mask_padded[k, i_pad - 1 : i_pad + 2, j_pad - 1 : j_pad + 2] = gaussian
            mosaic_feature_map_mask = mosaic_feature_map_mask_padded[:, 1:-1, 1:-1]
            mosaic_feature_map_mask = mosaic_feature_map_mask.unsqueeze(3).repeat(1, 1, 1, c)

            mosaic_fm_wo_cls_token = feature_map_spatial_repeated * mosaic_feature_map_mask
            mosaic_feature_map[:, 1:, :] = mosaic_fm_wo_cls_token.reshape(h * w, h * w, c)
        else:
            feature_map_repeated = feature_map.unsqueeze(0).repeat(h * w, 1, 1)
            mosaic_feature_map_mask = torch.zeros(h * w, num_tokens).to(feature_map.device)
            for i in range(h * w):
                mosaic_feature_map_mask[i, i + 1] = torch.ones(1).to(feature_map.device)
            if self._use_cls_token:
                mosaic_feature_map_mask[:, 0] = torch.ones(1).to(feature_map.device)
            mosaic_feature_map_mask = mosaic_feature_map_mask.unsqueeze(2).repeat(1, 1, c)
            mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask

        return mosaic_feature_map
