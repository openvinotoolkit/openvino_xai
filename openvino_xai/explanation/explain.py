# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import openvino.runtime as ov

import openvino_xai
from openvino_xai.algorithms.black_box.black_box_methods import RISE
from openvino_xai.common.parameters import TaskType
from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME, has_xai, logger
from openvino_xai.explanation.explanation_parameters import (
    ExplainMode,
    ExplanationParameters,
    TargetExplainGroup,
)
from openvino_xai.explanation.explanation_result import ExplanationResult
from openvino_xai.explanation.post_process import PostProcessor
from openvino_xai.explanation.utils import get_explain_target_indices
from openvino_xai.insertion.insertion_parameters import InsertionParameters


class Explainer:
    """
    Explainer sets explain mode, prepares the model, and generates explanations.

    Usage:
        explanation = explainer_object(data, explanation_parameters)

    :param model: Original model.
    :type model: ov.Model
    :param task_type: Type of the task.
    :type task_type: TaskType
    :param preprocess_fn: Preprocessing function.
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param postprocess_fn: Postprocessing functions, required for black-box.
    :type postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray]
    :param explain_mode: Explain mode.
    :type explain_mode: ExplainMode
    :param insertion_parameters: XAI insertion parameters.
    :type insertion_parameters: InsertionParameters]
    """

    def __init__(
        self,
        model: ov.Model,
        task_type: TaskType,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray] = None,
        explain_mode: ExplainMode = ExplainMode.AUTO,
        insertion_parameters: InsertionParameters | None = None,
    ) -> None:
        self.model = model
        self.compiled_model: ov.ie_api.CompiledModel | None = None
        self.task_type = task_type

        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        self.insertion_parameters = insertion_parameters

        self.explain_mode = explain_mode

        self._set_explain_mode()

        self._load_model()

    def _set_explain_mode(self) -> None:
        if self.explain_mode == ExplainMode.WHITEBOX:
            if has_xai(self.model):
                logger.info("Model already has XAI - using white-box mode.")
            else:
                self._insert_xai()
                logger.info("Explaining the model in the white-box mode.")
        elif self.explain_mode == ExplainMode.BLACKBOX:
            if self.postprocess_fn is None:
                raise ValueError("Postprocess function has to be provided for the black-box mode.")
            logger.info("Explaining the model in the black-box mode.")
        elif self.explain_mode == ExplainMode.AUTO:
            if has_xai(self.model):
                logger.info("Model already has XAI - using white-box mode.")
                self.explain_mode = ExplainMode.WHITEBOX
            else:
                try:
                    self._insert_xai()
                    self.explain_mode = ExplainMode.WHITEBOX
                    logger.info("Explaining the model in the white-box mode.")
                except Exception as e:
                    print(e)
                    logger.info("Failed to insert XAI into the model - use black-box mode.")
                    if self.postprocess_fn is None:
                        raise ValueError("Postprocess function has to be provided for the black-box mode.")
                    self.explain_mode = ExplainMode.BLACKBOX
                    logger.info("Explaining the model in the black-box mode.")
        else:
            raise ValueError(f"Not supported explain mode {self.explain_mode}.")

    def _insert_xai(self) -> None:
        logger.info("Model does not have XAI - trying to insert XAI to use white-box mode.")
        # Do we need to keep the original model?
        self.model = openvino_xai.insert_xai(self.model, self.task_type, self.insertion_parameters)

    def _load_model(self) -> None:
        self.compiled_model = ov.Core().compile_model(self.model, "CPU")

    def __call__(
        self,
        data: np.ndarray,
        explanation_parameters: ExplanationParameters,
        **kwargs,
    ) -> ExplanationResult:
        """Explainer call that generates processed explanation result."""
        if self.explain_mode == ExplainMode.WHITEBOX:
            saliency_map = self._generate_saliency_map_white_box(data)
        else:
            saliency_map = self._generate_saliency_map_black_box(data, explanation_parameters, **kwargs)

        explanation_result = ExplanationResult(
            saliency_map,
            explanation_parameters.target_explain_group,
            explanation_parameters.target_explain_labels,
            explanation_parameters.label_names,
        )
        explanation_result = PostProcessor(
            explanation=explanation_result,
            data=data,
            post_processing_parameters=explanation_parameters.post_processing_parameters,
        ).run()
        return explanation_result

    def model_forward(self, x: np.ndarray) -> ov.utils.data_helpers.wrappers.OVDict:
        """Forward pass of the compiled model. Will apply preprocess_fn, but not post-processing."""
        x = self.preprocess_fn(x)
        return self.compiled_model(x)

    def _generate_saliency_map_white_box(self, data: np.ndarray) -> np.ndarray:
        model_output = self.model_forward(data)
        return model_output[SALIENCY_MAP_OUTPUT_NAME]

    def _generate_saliency_map_black_box(
        self,
        data: np.ndarray,
        explanation_parameters: ExplanationParameters,
        **kwargs,
    ) -> np.ndarray:
        explain_target_indices = None
        if explanation_parameters.target_explain_group == TargetExplainGroup.CUSTOM:
            explain_target_indices = get_explain_target_indices(
                explanation_parameters.target_explain_labels,
                explanation_parameters.label_names,
            )
        if self.task_type == TaskType.CLASSIFICATION:
            saliency_map = RISE.run(
                self.compiled_model,
                self.preprocess_fn,
                self.postprocess_fn,
                data,
                explain_target_indices,
                **kwargs,
            )
            return saliency_map
        raise ValueError(f"Task type {self.task_type} is not supported in the black-box mode.")
