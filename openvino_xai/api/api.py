# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, TypeVar

import openvino as ov

from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import IdentityPreprocessFN, has_xai, logger
from openvino_xai.methods.factory import WhiteBoxMethodFactory
from openvino_xai.utils.torch import torch

Model = TypeVar("Model", ov.Model, torch.nn.Module)


def insert_xai(
    model: Model,
    task: Task,
    explain_method: Method | None = None,
    target_layer: str | List[str] | None = None,
    embed_scaling: bool | None = True,
    **kwargs,
) -> Model:
    """
    Inserts XAI branch into the given model.

    Usage:
        model_xai = openvino_xai.insert_xai(model, task=Task.CLASSIFICATION)

    :param model: Original model.
    :type model: ov.Model | torch.nn.Module
    :param task: Type of the task: CLASSIFICATION or DETECTION.
    :type task: Task
    :parameter explain_method: Explain method to use for model explanation.
    :type explain_method: Method
    :parameter target_layer: Target layer(s) (node(s)) name after which the XAI branch will be inserted.
    :type target_layer: str | List[str]
    :parameter embed_scaling: If set to True, saliency map scale (0 ~ 255) operation is embedded in the model.
    :type embed_scaling: bool
    """

    if has_xai(model):
        logger.info("Provided model already contains XAI branch, return it as-is.")
        return model

    method = WhiteBoxMethodFactory.create_method(
        task=task,
        model=model,
        preprocess_fn=IdentityPreprocessFN(),
        explain_method=explain_method,
        target_layer=target_layer,
        embed_scaling=embed_scaling,
        **kwargs,
    )

    model_xai = method.prepare_model(load_model=False)

    if not has_xai(model_xai):
        raise RuntimeError("Insertion of the XAI branch into the model was not successful.")
    logger.info("Insertion of the XAI branch into the model was successful.")

    return model_xai
