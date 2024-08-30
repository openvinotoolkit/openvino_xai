# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino as ov
from openvino.preprocess import PrePostProcessor

from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME


def insert_xai_branch_into_model(
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
