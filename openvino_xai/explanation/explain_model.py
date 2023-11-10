# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Union

import numpy as np
import openvino.model_api as mapi

from openvino_xai.explanation.explanation_result import ExplanationResult
from openvino_xai.explanation.explainers import WhiteBoxExplainer, BlackBoxExplainer
from openvino_xai.explanation.explanation_parameters import ExplainMode, ExplanationParameters
from openvino_xai.explanation.utils import InferenceResult


def explain(
        model_inferrer: Union[Callable[[np.ndarray], InferenceResult], mapi.models.Model],
        data: np.ndarray,
        explanation_parameters: ExplanationParameters = ExplanationParameters(),
) -> ExplanationResult:
    """
    Explains the model-data pair, i.e. generates explanation result.

    :param model_inferrer: Callable model inferrer object.
    :type model_inferrer: Union[Callable[[np.ndarray], InferenceResult], mapi.models.Model]
    :param data: Data to explain.
    :type data: np.ndarray
    :param explanation_parameters: Explanation parameters.
    :type explanation_parameters: ExplanationParameters
    :return: Explanation result object, that contain saliency map and other useful info.
    """

    if explanation_parameters.explain_mode == ExplainMode.WHITEBOX:
        explainer = WhiteBoxExplainer(model_inferrer)
    elif explanation_parameters.explain_mode == ExplainMode.BLACKBOX:
        explainer = BlackBoxExplainer(model_inferrer)
    else:
        raise ValueError(f"Explain mode {explanation_parameters.explain_mode} is not supported.")

    explanation = explainer.explain(
        data,
        explanation_parameters,
    )
    return explanation
