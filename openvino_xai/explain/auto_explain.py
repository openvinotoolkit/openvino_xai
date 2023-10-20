# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import numpy as np
import openvino

from openvino_xai.explain.base import Explainer
from openvino_xai.explain.black_box import RISEExplainer
from openvino_xai.explain.white_box import WhiteBoxExplainer

from openvino_xai.model import XAIModel, XAIClassificationModel
from openvino_xai.parameters import ExplainParameters, PostProcessParameters

from openvino_xai.saliency_map import ExplainResult, TargetExplainGroup
from openvino_xai.utils import logger



class AutoExplainer(Explainer):
    """Base class for auto-explainers, using white box or black box approach.

    :param model: ModelAPI wrapper.
    :type model: openvino.model_api.models.Model
    :param explain_parameters: Explain parameters.
    :type explain_parameters: Optional[ExplainParameters]
    """

    def __init__(self, model: openvino.model_api.models.Model, explain_parameters: Optional[ExplainParameters] = None):
        super().__init__(model)
        self._explain_parameters = explain_parameters


class ClassificationAutoExplainer(AutoExplainer):
    """Explain classification models in auto mode, using white box or black box approach."""

    def explain(
        self,
        data: np.ndarray,
        target_explain_group: Optional[TargetExplainGroup] = None,
        explain_targets: Optional[List[int]] = None,
        post_processing_parameters: PostProcessParameters = PostProcessParameters(),
    ) -> ExplainResult:
        """
        Implements three explain scenarios, for different IR models:
            1. IR model contain xai branch -> infer Model API wrapper.
            2. If not (1), IR model can be augmented with XAI branch -> augment and infer.
            3. If not (1) and (2), IR model can NOT be augmented with XAI branch -> use XAI BB method.

        :param data: Data to explain.
        :type data: np.ndarray
        :param target_explain_group: Target explain group.
        :type target_explain_group: TargetExplainGroup
        :param explain_targets: Provides list of custom targets, optional.
        :type explain_targets: Optional[List[int]]
        :param post_processing_parameters: Parameters that define post-processing.
        :type post_processing_parameters: PostProcessParameters
        """
        extra_params = {"target_explain_group": target_explain_group,
                        "explain_targets": explain_targets,
                        "post_processing_parameters": post_processing_parameters}

        if XAIModel.has_xai(self._model.inference_adapter.model):
            logger.info("Model already has XAI - using White Box explainer.")
            explanations = WhiteBoxExplainer(self._model).explain(data, **extra_params)
            return explanations
        else:
            try:
                logger.info("Model does not have XAI - trying to insert XAI and use White Box explainer.")
                self._model = XAIClassificationModel.insert_xai(self._model, self._explain_parameters)
                explanations = WhiteBoxExplainer(self._model).explain(data, **extra_params)
                return explanations
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model. Calling Black Box explainer.")
                explanations = RISEExplainer(self._model).explain(data, **extra_params)
                return explanations


class DetectionAutoExplainer(AutoExplainer):
    """Explain detection models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray) -> np.ndarray:
        """Explain the input."""
        raise NotImplementedError
