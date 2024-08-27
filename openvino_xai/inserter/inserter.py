# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence, TypeAlias

import openvino as ov
import torch
from openvino.preprocess import PrePostProcessor

from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME

FeatureMapType: TypeAlias = torch.Tensor | Sequence[torch.Tensor]
HeadForwardFn: TypeAlias = Callable[[FeatureMapType], torch.Tensor]
ExplainerForwardFn: TypeAlias = HeadForwardFn


def insert_xai_branch_into_ov_model(
    model: ov.Model,
    xai_output_node: ov.runtime.Node,
    set_uint8: bool,
) -> ov.Model:
    """Create new model with XAI branch."""
    model_ori_outputs = model.outputs
    model_ori_params = model.get_parameters()
    model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

    xai_output_index = len(model_ori_outputs)
    model_xai = _set_xai_output_name_and_precision(model_xai, xai_output_index, set_uint8)
    return model_xai


def _set_xai_output_name_and_precision(
    model_xai: ov.Model,
    xai_output_index: int,
    set_uint8: bool,
) -> ov.Model:
    model_xai.outputs[xai_output_index].tensor.set_names({SALIENCY_MAP_OUTPUT_NAME})
    if set_uint8:
        ppp = PrePostProcessor(model_xai)
        ppp.output(SALIENCY_MAP_OUTPUT_NAME).tensor().set_element_type(ov.Type.u8)
        model_xai = ppp.build()
    return model_xai


def insert_xai_branch_into_torch_model(
    model: torch.nn.Module,
    feature_module: torch.nn.Module | str,
    xai_module: torch.nn.Module,
    # set_uint8: bool,
) -> torch.nn.Module:
    """Insert XAI functionality to given model via module hooks."""
    model.register_module(
        name=SALIENCY_MAP_OUTPUT_NAME,
        module=xai_module,
    )

    def xai_hook(module, args, output):
        """Generate saliency map from feature map."""
        xai_output = xai_module(output)
        xai_module.register_buffer(
            name=SALIENCY_MAP_OUTPUT_NAME,
            tensor=xai_output,
        )

    if isinstance(feature_module, str):
        feature_module = model.get_submodule(feature_module)
    feature_module.register_forward_hook(xai_hook)

    def output_hook(module, args, output):
        """Pack outputs with model prediction and additional saliency map."""
        return {"prediction": output, SALIENCY_MAP_OUTPUT_NAME: xai_module.get_buffer(SALIENCY_MAP_OUTPUT_NAME)}

    model.register_forward_hook(output_hook)

    return model
