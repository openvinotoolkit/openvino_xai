# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copy & edit from https://github.com/openvinotoolkit/training_extensions/blob/2.1.0/tests/unit/algo/explain/test_xai_algorithms.py

from typing import Any, Callable, Dict, Mapping, Sequence, TypeAlias

import numpy as np
import torch

from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME, has_xai
from openvino_xai.methods.white_box.torch import (  # DetClassProbabilityMap,; ReciproCAM,; ViTReciproCAM,
    ActivationMap,
    TorchMethod,
)


def test_normalize():
    x = torch.rand((2, 2)) * 100
    y = TorchMethod._normalize_map(x)
    assert x.shape == y.shape
    assert torch.all(y >= 0)
    assert torch.all(y <= 255)
    x = torch.rand((2, 2, 2)) * 100
    y = TorchMethod._normalize_map(x)
    assert x.shape == y.shape
    assert torch.all(y >= 0)
    assert torch.all(y <= 255)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Identity()
        self.neck = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.output = torch.nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        x = self.neck(x)
        return self.output(x)


def test_torch_method():
    model = DummyModel()
    method = TorchMethod(model=model, target_layer="feature")
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.zeros((1, 3, 5, 5))
    output = method.model_forward(data)
    assert type(output) == dict
    assert SALIENCY_MAP_OUTPUT_NAME in output

    class DummyMethod(TorchMethod):
        def _feature_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            output = torch.cat((output, output), dim=0)
            return super()._feature_hook(module, inputs, output)

        def _output_hook(
            self, module: torch.nn.Module, inputs: Any, output: torch.Tensor
        ) -> Dict[str, torch.Tensor | None]:
            return {
                "prediction": output[0:],
                SALIENCY_MAP_OUTPUT_NAME: output[1:],
            }

    model = DummyModel()
    method = DummyMethod(model=model, target_layer="feature")
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(1, 3, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    prediction = output["prediction"]
    saliency_maps = output[SALIENCY_MAP_OUTPUT_NAME]
    assert np.all(saliency_maps == prediction)


def test_activationmap() -> None:
    model = DummyModel()
    method = ActivationMap(model=model, target_layer="feature")
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(1, 3, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    saliency_maps = output[SALIENCY_MAP_OUTPUT_NAME]
    assert saliency_maps.shape == torch.Size([1, 5, 5])
    assert np.all(saliency_maps >= 0)
    assert np.all(saliency_maps <= 255)
    assert saliency_maps.dtype == np.uint8


# def test_reciprocam() -> None:
#     def cls_head_forward_fn(_) -> None:
#         return torch.zeros((25, 2))
#
#     num_classes = 2
#     optimize_gap = False
#     explain_algo = ReciproCAM(
#         cls_head_forward_fn,
#         num_classes=num_classes,
#         optimize_gap=optimize_gap,
#     )
#
#     assert explain_algo._norm_saliency_maps
#
#     feature_map = torch.zeros((1, 10, 5, 5))
#
#     saliency_maps = explain_algo.func(feature_map)
#     assert saliency_maps.size() == torch.Size([1, 2, 5, 5])
#
#
# def test_vitreciprocam() -> None:
#     def cls_head_forward_fn(_) -> None:
#         return torch.zeros((196, 2))
#
#     num_classes = 2
#     explain_algo = ViTReciproCAM(
#         cls_head_forward_fn,
#         num_classes=num_classes,
#     )
#
#     assert explain_algo._norm_saliency_maps
#
#     feature_map = torch.zeros((1, 197, 192))
#
#     saliency_maps = explain_algo.func(feature_map)
#     assert saliency_maps.size() == torch.Size([1, 2, 14, 14])
#
#
# def test_detclassprob() -> None:
#     num_classes = 2
#     num_anchors = [1] * 10
#     explain_algo = DetClassProbabilityMap(
#         num_classes=num_classes,
#         num_anchors=num_anchors,
#     )
#
#     assert explain_algo._norm_saliency_maps
#
#     backbone_out = torch.zeros((1, 5, 2, 2, 2))
#
#     saliency_maps = explain_algo.func(backbone_out)
#     assert saliency_maps.size() == torch.Size([5, 2, 2, 2])
