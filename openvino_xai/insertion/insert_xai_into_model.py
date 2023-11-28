# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Union, Optional

import openvino
import openvino.runtime as ov
from openvino.runtime import Type
from openvino.preprocess import PrePostProcessor
import openvino.model_api as mapi

from openvino_xai.algorithms.white_box.create_method import create_white_box_classification_explain_method, \
    create_white_box_detection_explain_method
from openvino_xai.common.parameters import TaskType
from openvino_xai.insertion.insertion_parameters import InsertionParameters
from openvino_xai.common.utils import logger, has_xai, SALIENCY_MAP_OUTPUT_NAME


def insert_xai(
        model: Union[ov.Model, str],
        task_type: TaskType,
        insertion_parameters: Optional[InsertionParameters] = None,
) -> ov.Model:
    """
    Inserts XAI into IR.

    :param model: Original IR or path to .xml.
    :type model: Union[ov.Model, str]
    :param task_type: Type of the model.
    :type task_type: TaskType
    :param insertion_parameters: Insertion parameters that parametrize white-box method,
        that will be inserted into the model graph.
    :type insertion_parameters: InsertionParameters
    :return: IR with XAI head.
    """

    if isinstance(model, str):
        model_suffix = Path(model).suffix
        if not model_suffix == ".xml":
            raise ValueError(f"XAI can be inserted only into OV IR, "
                             f"but provided model has {model_suffix} extension. "
                             f"Please provide path to OV IR for white-box explanation methods.")
        model = openvino.runtime.Core().read_model(model)

    if has_xai(model):
        logger.info("Provided IR model already contains XAI branch, return it as-is.")
        return model

    model_xai = _insert_xai_branch_into_model(model, task_type, insertion_parameters)

    if not has_xai(model_xai):
        raise RuntimeError("Insertion of the XAI branch into the model was not successful.")
    logger.info("Insertion of the XAI branch into the model was successful.")

    return model_xai


def _insert_xai_branch_into_model(
        model: ov.Model,
        task_type: TaskType,
        insertion_parameters: Optional[InsertionParameters]
) -> ov.Model:
    if task_type == TaskType.CLASSIFICATION:
        explain_method = create_white_box_classification_explain_method(model, insertion_parameters)
    elif task_type == TaskType.DETECTION:
        explain_method = create_white_box_detection_explain_method(model, insertion_parameters)
    else:
        raise ValueError(f"Model type {task_type} is not supported")

    xai_output_node = explain_method.generate_xai_branch()
    model_ori_outputs = model.outputs
    model_ori_params = model.get_parameters()
    model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

    xai_output_index = len(model_ori_outputs)
    set_uint8 = explain_method.embed_normalization  # TODO: make a property
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
        ppp.output(SALIENCY_MAP_OUTPUT_NAME).tensor().set_element_type(Type.u8)
        model_xai = ppp.build()
    return model_xai


def insert_xai_into_mapi_wrapper(
        mapi_wrapper: mapi.models.Model,
        insertion_parameters: Optional[InsertionParameters] = None,
) -> mapi.models.Model:
    """
    Insert XAI into IR stored in Model API wrapper.

    :param mapi_wrapper: Original ModelAPI wrapper.
    :type mapi_wrapper: openvino.model_api.models.Model
    :param insertion_parameters: Insertion parameters that parametrize white-box method,
        that will be inserted into the model graph.
    :type insertion_parameters: InsertionParameters
    :return: Modified ModelAPI wrapper with XAI head.
    """

    model = mapi_wrapper.get_model()

    if isinstance(mapi_wrapper, openvino.model_api.models.ClassificationModel):
        task_type = TaskType.CLASSIFICATION
    elif isinstance(mapi_wrapper, openvino.model_api.models.DetectionModel):
        task_type = TaskType.DETECTION
    else:
        raise ValueError(f"Model type {type(mapi_wrapper)} is not supported.")

    model_xai = insert_xai(model, task_type=task_type, insertion_parameters=insertion_parameters)

    # Update Model API wrapper
    mapi_wrapper.inference_adapter.model = model_xai
    if hasattr(mapi_wrapper, "out_layer_names"):
        mapi_wrapper.out_layer_names.append(SALIENCY_MAP_OUTPUT_NAME)
    if mapi_wrapper.model_loaded:
        mapi_wrapper.load(force=True)
    return mapi_wrapper
