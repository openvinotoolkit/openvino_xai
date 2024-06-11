# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime as ov

from openvino_xai.common.parameters import Task
from openvino_xai.common.utils import IdentityPreprocessFN, has_xai, logger
from openvino_xai.inserter.parameters import InsertionParameters
from openvino_xai.methods.factory import WhiteBoxMethodFactory


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
