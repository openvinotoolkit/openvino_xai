# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copy & edit from https://github.com/openvinotoolkit/training_extensions/blob/2.1.0/tests/unit/algo/explain/test_xai_algorithms.py

from typing import Any, Callable, Dict, Mapping, Sequence, TypeAlias

import numpy as np
import pytest
import torch

from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME, has_xai
from openvino_xai.methods.white_box.torch import (
    TorchActivationMap,
    TorchReciproCAM,
    TorchViTReciproCAM,
    TorchWhiteBoxMethod,
)


def test_normalize():
    x = torch.rand((2, 2)) * 100
    y = TorchWhiteBoxMethod._normalize_map(x)
    assert x.shape == y.shape
    assert torch.all(y >= 0)
    assert torch.all(y <= 255)
    x = torch.rand((2, 2, 2)) * 100
    y = TorchWhiteBoxMethod._normalize_map(x)
    assert x.shape == y.shape
    assert torch.all(y >= 0)
    assert torch.all(y <= 255)


class DummyCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.feature = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
        )
        self.neck = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.output = torch.nn.LazyLinear(out_features=num_classes)

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        x = self.neck(x)
        x = x.view(x.shape[0], -1)
        x = self.output(x)
        return torch.nn.functional.softmax(x, dim=1)


class DummyVIT(torch.nn.Module):
    def __init__(self, num_classes: int = 2, dim: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.pre = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Identity(),
        )
        self.feature = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
        )
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)
        self.output = torch.nn.LazyLinear(out_features=num_classes)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self.pre(x)
        x = x.reshape(b, c, h * w)
        x = x.transpose(1, 2)
        x = torch.cat([torch.rand((b, 1, c)), x], dim=1)
        x = self.feature(x)
        x = x + self.norm1(x)
        x = x + self.norm2(x)
        x = x + self.norm3(x)
        x = self.output(x[:, 0])
        return torch.nn.functional.softmax(x, dim=1)


def test_torch_method():
    model = DummyCNN()

    with pytest.raises(ValueError):
        method = TorchWhiteBoxMethod(model=model, target_layer="something_else")
        model_xai = method.prepare_model()

    method = TorchWhiteBoxMethod(model=model, target_layer="feature")
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.zeros((1, 3, 5, 5))
    output = method.model_forward(data)
    assert type(output) == dict
    assert SALIENCY_MAP_OUTPUT_NAME in output

    class DummyMethod(TorchWhiteBoxMethod):
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

    model = DummyCNN()
    method = DummyMethod(model=model, target_layer="feature")
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(1, 3, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    prediction = output["prediction"]
    saliency_maps = output[SALIENCY_MAP_OUTPUT_NAME]
    assert np.all(saliency_maps == prediction)


def test_prepare_model():
    model = DummyCNN()
    method = TorchWhiteBoxMethod(model=model, target_layer="feature", prepare_model=False)
    model_xai = method.prepare_model(load_model=False)
    assert method._model_compiled is None
    model_xai = method.prepare_model(load_model=False)
    assert method._model_compiled is None
    assert model is not model_xai

    model_xai = method.prepare_model(load_model=True)
    assert method._model_compiled is not None
    assert model is not model_xai

    model.has_xai = True
    method = TorchWhiteBoxMethod(model=model, target_layer="feature")
    model_xai = method.prepare_model(load_model=False)
    assert model_xai == model


def test_detect_feature_layer():
    model = DummyCNN()
    method = TorchWhiteBoxMethod(model=model, target_layer=None)
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(1, 3, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    assert method._feature_module is model_xai.feature
    output = method.model_forward(data)
    assert type(output) == dict  # still good for 2nd forward

    model = DummyVIT()
    with pytest.raises(RuntimeError):
        # 4D feature map search should fail for ViTs
        method = TorchWhiteBoxMethod(model=model, target_layer=None)

    model = DummyVIT()
    method = TorchViTReciproCAM(model=model, target_layer=None)
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(1, 3, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    assert method._feature_module is model_xai.norm1
    output = method.model_forward(data)
    assert type(output) == dict  # still good for 2nd forward


def test_activationmap() -> None:
    batch_size = 2
    num_classes = 3
    model = DummyCNN(num_classes=num_classes)
    method = TorchActivationMap(model=model, target_layer="feature")
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(batch_size, 3, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    saliency_maps = output[SALIENCY_MAP_OUTPUT_NAME]
    assert saliency_maps.shape == torch.Size([batch_size, 5, 5])
    assert np.all(saliency_maps >= 0)
    assert np.all(saliency_maps <= 255)
    assert saliency_maps.dtype == np.uint8


@pytest.mark.parametrize("optimize_gap", [True, False])
def test_reciprocam(optimize_gap: bool) -> None:
    batch_size = 2
    num_classes = 3
    model = DummyCNN(num_classes=num_classes)
    method = TorchReciproCAM(model=model, target_layer="feature", optimize_gap=optimize_gap)
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(batch_size, 4, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    saliency_maps = output[SALIENCY_MAP_OUTPUT_NAME]
    assert saliency_maps.shape == torch.Size([batch_size, num_classes, 5, 5])
    assert np.all(saliency_maps >= 0)
    assert np.all(saliency_maps <= 255)
    assert saliency_maps.dtype == np.uint8


@pytest.mark.parametrize("use_gaussian", [True, False])
@pytest.mark.parametrize("use_cls_token", [True, False])
def test_vitreciprocam(use_gaussian: bool, use_cls_token: bool) -> None:
    batch_size = 2
    num_classes = 3
    dim = 3
    model = DummyVIT(num_classes=num_classes, dim=dim)
    method = TorchViTReciproCAM(
        model=model, target_layer="feature", use_gaussian=use_gaussian, use_cls_token=use_cls_token
    )
    model_xai = method.prepare_model()
    assert has_xai(model_xai)
    data = np.random.rand(batch_size, dim, 5, 5)
    output = method.model_forward(data)
    assert type(output) == dict
    saliency_maps = output[SALIENCY_MAP_OUTPUT_NAME]
    assert saliency_maps.shape == torch.Size([batch_size, num_classes, 5, 5])
    assert np.all(saliency_maps >= 0)
    assert np.all(saliency_maps <= 255)
    assert saliency_maps.dtype == np.uint8
