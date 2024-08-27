# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from openvino_xai.inserter.inserter import insert_xai_branch_into_torch_model
from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME


def test_insert_xai_branch_into_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = torch.nn.Identity()
            self.output = torch.nn.Identity()
        def forward(self, x: torch.Tensor):
            x = self.feature(x)
            return self.output(x)

    class XAI(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return x + 1

    # With module
    model = Model()
    xai_module = XAI()
    data = torch.zeros((1, 3, 10, 10))

    model_xai = insert_xai_branch_into_torch_model(
        model=model,
        feature_module=model.feature,
        xai_module=xai_module,
    )
    assert model.get_submodule(SALIENCY_MAP_OUTPUT_NAME) is xai_module
    output = model_xai(data)
    assert type(output) == dict
    assert SALIENCY_MAP_OUTPUT_NAME in output
    assert torch.equal(output[SALIENCY_MAP_OUTPUT_NAME], data + 1)

    # With name
    model = Model()
    xai_module = XAI()
    data = torch.zeros((1, 3, 10, 10))

    model_xai = insert_xai_branch_into_torch_model(
        model=model,
        feature_module="feature",
        xai_module=xai_module,
    )
    assert model.get_submodule(SALIENCY_MAP_OUTPUT_NAME) is xai_module
    output = model_xai(data)
    assert type(output) == dict
    assert SALIENCY_MAP_OUTPUT_NAME in output
    assert torch.equal(output[SALIENCY_MAP_OUTPUT_NAME], data + 1)
