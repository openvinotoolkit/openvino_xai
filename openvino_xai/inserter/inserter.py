# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor

from openvino_xai import Task
from openvino_xai.common.utils import (
    SALIENCY_MAP_OUTPUT_NAME,
    IdentityPreprocessFN,
    has_xai,
    logger,
)
from openvino_xai.inserter.parameters import InsertionParameters


def insert_xai(
    model: ov.Model,
    task: Task,
    insertion_parameters: InsertionParameters | None = None,
) -> ov.Model:
    """
    Function that inserts XAI branch into IR.

    Usage:
        model_xai = openvino_xai.insert_xai(model, task=Task.CLASSIFICATION)

    :param model: Original IR.
    :type model: ov.Model | str
    :param task: Type of the task: CLASSIFICATION or DETECTION.
    :type task: Task
    :param insertion_parameters: Insertion parameters that parametrize white-box method,
        that will be inserted into the model graph (optional).
    :type insertion_parameters: InsertionParameters
    :return: IR with XAI branch.
    """
    from openvino_xai.methods.factory import WhiteBoxMethodFactory

    if has_xai(model):
        logger.info("Provided IR model already contains XAI branch, return it as-is.")
        return model

    method = WhiteBoxMethodFactory.create_method(
        task=task,
        model=model,
        preprocess_fn=IdentityPreprocessFN(),
        insertion_parameters=insertion_parameters,
        prepare_model=False,
    )

    model_xai = method.prepare_model(load_model=False)

    if not has_xai(model_xai):
        raise RuntimeError("Insertion of the XAI branch into the model was not successful.")
    logger.info("Insertion of the XAI branch into the model was successful.")

    return model_xai


def insert_xai_branch_into_model(
    model: ov.Model,
    xai_output_node,
    set_uint8,
) -> ov.Model:
    """Creates new model with XAI branch."""
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
