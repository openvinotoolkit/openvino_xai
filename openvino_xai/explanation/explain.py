# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import openvino.runtime as ov

from openvino_xai.algorithms.black_box.black_box_methods import BlackBoxXAIMethodBase
from openvino_xai.algorithms.create_method import BlackBoxMethodFactory, WhiteBoxMethodFactory
from openvino_xai.common.parameters import TaskType
from openvino_xai.common.utils import (
    IdentityPreprocessFN,
    logger,
)
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
    Explainer creates methods uses them to generate explanations.

    Usage:
        explanation = explainer_object(data, explanation_parameters)

    :param model: Original model.
    :type model: ov.Model
    :param task_type: Type of the task.
    :type task_type: TaskType
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
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
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray] = None,
        explain_mode: ExplainMode = ExplainMode.AUTO,
        insertion_parameters: InsertionParameters | None = None,
    ) -> None:
        self.model = model
        self.compiled_model: ov.ie_api.CompiledModel | None = None
        self.task_type = task_type

        if isinstance(preprocess_fn, IdentityPreprocessFN):
            logger.info(
                "Assigning preprocess_fn to identity function assumes that input images were already preprocessed "
                "by user before passing it to the model. "
                "Please define preprocessing function OR preprocess images beforehand."
            )
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        self.insertion_parameters = insertion_parameters

        self.explain_mode = explain_mode

        self.create_method(self.explain_mode, self.task_type)

    def create_method(self, explain_mode: ExplainMode, task_type: TaskType) -> None:
        if explain_mode == ExplainMode.WHITEBOX:
            try:
                self.method = WhiteBoxMethodFactory.create_method(task_type, self.model, self.preprocess_fn, self.insertion_parameters)
                logger.info("Explaining the model in white-box mode.")
            except Exception as e:
                print(e)
                raise RuntimeError("Failed to insert XAI into the model. Try to use black-box.")
        elif self.explain_mode == ExplainMode.BLACKBOX:
            self._check_postprocess_fn()
            self.method = BlackBoxMethodFactory.create_method(task_type, self.model, self.preprocess_fn, self.postprocess_fn)
        elif self.explain_mode == ExplainMode.AUTO:
            try:
                self.method = WhiteBoxMethodFactory.create_method(task_type, self.model, self.preprocess_fn, self.insertion_parameters)
                logger.info("Explaining the model in the white-box mode.")
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model - using black-box mode.")
                self._check_postprocess_fn()
                self.method = BlackBoxMethodFactory.create_method(task_type, self.model, self.preprocess_fn, self.postprocess_fn)
                logger.info("Explaining the model in the black-box mode.")
        else:
            raise ValueError(f"Not supported explain mode {self.explain_mode}.")

    def __call__(
        self,
        data: np.ndarray,
        explanation_parameters: ExplanationParameters,
        **kwargs,
    ) -> ExplanationResult:
        """Explainer call that generates processed explanation result."""
        explain_target_indices = None
        if isinstance(self.method, BlackBoxXAIMethodBase) and explanation_parameters.target_explain_group == TargetExplainGroup.CUSTOM:
            explain_target_indices = get_explain_target_indices(
                explanation_parameters.target_explain_labels,
                explanation_parameters.label_names,
            )

        saliency_map = self.method.generate_saliency_map(
            data,
            explain_target_indices=explain_target_indices,
            **kwargs,
        )

        explanation_result = ExplanationResult(
            saliency_map=saliency_map,
            target_explain_group=explanation_parameters.target_explain_group,
            target_explain_labels=explanation_parameters.target_explain_labels,
            label_names=explanation_parameters.label_names,
        )
        return self._post_process(explanation_result, data, explanation_parameters)

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> ov.utils.data_helpers.wrappers.OVDict:
        """Forward pass of the compiled model."""
        if self.method.model_compiled is None:
            raise RuntimeError("Model is not compiled. Call prepare_model() first.")
        return self.method.model_forward(x, preprocess)

    def _post_process(self, explanation_result, data, explanation_parameters):
        if not isinstance(self.preprocess_fn, IdentityPreprocessFN):
            # Assume if preprocess_fn is provided - input data is original image
            explanation_result = PostProcessor(
                explanation=explanation_result,
                original_input_image=data,
                post_processing_parameters=explanation_parameters.post_processing_parameters,
            ).run()
        else:
            # preprocess_fn is not provided - assume input data is processed
            explanation_result = PostProcessor(
                explanation=explanation_result,
                output_size=data.shape[:2],  # resize to model input by default
                post_processing_parameters=explanation_parameters.post_processing_parameters,
            ).run()
        return explanation_result

    def _check_postprocess_fn(self) -> None:
        if self.postprocess_fn is None:
            raise ValueError("Postprocess function has to be provided for the black-box mode.")
