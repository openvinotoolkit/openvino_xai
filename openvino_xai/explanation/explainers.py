# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Union, Callable

import numpy as np
import openvino.model_api as mapi

from openvino_xai.explanation.explanation_parameters import ExplanationParameters
from openvino_xai.explanation.utils import InferenceResult
from openvino_xai.explanation.explanation_result import ExplanationResult
from openvino_xai.explanation.post_process import PostProcessor
from openvino_xai.algorithms.black_box.black_box_methods import RISE
from openvino_xai.common.parameters import XAIMethodType


class Explainer(ABC):
    """A base interface for explainer.

    :param model_inferrer: Callable model inferrer object.
    :type model_inferrer: Union[Callable[[np.ndarray], InferenceResult], mapi.models.Model]
    """

    def __init__(self, model_inferrer: Union[Callable[[np.ndarray], InferenceResult], mapi.models.Model]):
        self._model_inferrer = model_inferrer

    def explain(
            self,
            data: np.ndarray,
            explanation_parameters: ExplanationParameters = ExplanationParameters(),
    ) -> ExplanationResult:
        """
        Explains the data, i.e. generates explanation result.

        :param data: Data to explanation.
        :type data: np.ndarray
        :param explanation_parameters: Explanation parameters.
        :type explanation_parameters: ExplanationParameters
        :return: Explanation result object, that contain saliency map and other useful info.
        """
        # TODO: handle path_to_data as input as well?
        inference_result = self.get_inference_result(data, explanation_parameters)

        explanation_result = ExplanationResult(
            inference_result,
            explanation_parameters.target_explain_group,
            explanation_parameters.explain_target_names,
            explanation_parameters.custom_target_indices,
            explanation_parameters.confidence_threshold,
        )
        explanation_result = PostProcessor(
            explanation_result,
            data,
            explanation_parameters.post_processing_parameters,
        ).postprocess()
        return explanation_result

    @abstractmethod
    def get_inference_result(self, *args, **kwargs):
        """Generates inference result."""
        raise NotImplementedError


class WhiteBoxExplainer(Explainer):
    """White Box explainer explains the model with XAI branch injected."""

    def get_inference_result(self, data: np.ndarray, _) -> Union[InferenceResult, mapi.models.ClassificationResult]:
        """Generates inference result in white box mode.

        :param data: Data to explanation.
        :type data: np.ndarray
        :return: Inference result.
        """
        inference_result = self._model_inferrer(data)
        return inference_result


class BlackBoxExplainer(Explainer):
    """Black Box explainer explains the model as a black-box."""

    def get_inference_result(
            self, data: np.ndarray, explanation_parameters: ExplanationParameters
    ) -> Union[InferenceResult, mapi.models.ClassificationResult]:
        """Generates inference result in black box mode.

        :param data: Data to explanation.
        :type data: np.ndarray
        :param explanation_parameters: Explanation parameters.
        :type explanation_parameters: ExplanationParameters
        :return: Inference result.
        """
        if explanation_parameters.black_box_method == XAIMethodType.RISE:
            black_box_method = RISE(self._model_inferrer, **explanation_parameters.black_box_method_kwargs)
        else:
            raise ValueError(f"{explanation_parameters.black_box_method} is not supported.")

        inference_result = black_box_method.get_result(data, explanation_parameters)
        return inference_result
