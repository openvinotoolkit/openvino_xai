# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import openvino

from openvino_xai.explain import Explainer, RISEExplainer, WhiteBoxExplainer
from openvino_xai.model import XAIModel, XAIClassificationModel
from openvino_xai.parameters import ExplainParameters

from openvino_xai.saliency_map import ExplainResult, TargetExplainGroup
from openvino_xai.utils import logger



class AutoExplainer(Explainer):
    """Explain in auto mode, using white box or black box approach."""

    def __init__(self, model: openvino.model_api.models.Model, explain_parameters: Optional[ExplainParameters] = None):
        super().__init__(model)
        self._explain_parameters = explain_parameters


class ClassificationAutoExplainer(AutoExplainer):
    """Explain classification models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray, target_explain_group: Optional[TargetExplainGroup] = None) -> ExplainResult:
        """
        Implements three explain scenarios, for different IR models:
            1. IR model contain xai branch -> infer Model API wrapper.
            2. If not (1), IR model can be augmented with XAI branch -> augment and infer.
            3. If not (1) and (2), IR model can NOT be augmented with XAI branch -> use XAI BB method.

        :param data: Data to explain.
        :type data: np.ndarray
        :param target_explain_group: Target explain group.
        :type target_explain_group: TargetExplainGroup
        """
        if XAIModel.has_xai(self._model.inference_adapter.model):
            logger.info("Model already has XAI - using White Box explainer.")
            explanations = WhiteBoxExplainer(self._model).explain(data, target_explain_group)
            return explanations
        else:
            try:
                logger.info("Model does not have XAI - trying to insert XAI and use White Box explainer.")
                self._model = XAIClassificationModel.insert_xai(self._model, self._explain_parameters)
                explanations = WhiteBoxExplainer(self._model).explain(data)
                return explanations
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model. Calling Black Box explainer.")
                explanations = RISEExplainer(self._model).explain(data)
                return explanations


class DetectionAutoExplainer(AutoExplainer):
    """Explain detection models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray) -> np.ndarray:
        """Explain the input."""
        raise NotImplementedError