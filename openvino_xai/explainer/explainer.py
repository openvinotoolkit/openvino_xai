# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from openvino_xai import Task
from openvino_xai.common.utils import IdentityPreprocessFN, logger
from openvino_xai.explainer.explanation import Explanation
from openvino_xai.explainer.parameters import (
    ExplainMode,
    ExplanationParameters,
    TargetExplainGroup,
)
from openvino_xai.explainer.utils import get_explain_target_indices
from openvino_xai.explainer.visualizer import Visualizer
from openvino_xai.inserter.parameters import InsertionParameters
from openvino_xai.methods.base import MethodBase
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod
from openvino_xai.methods.factory import BlackBoxMethodFactory, WhiteBoxMethodFactory


class Explainer:
    """
    Explainer creates methods and uses them to generate explanations.

    Usage:
        explanation = explainer_object(data, explanation_parameters)

    :param model: Original model.
    :type model: ov.Model
    :param task: Type of the task: CLASSIFICATION or DETECTION.
    :type task: Task
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param postprocess_fn: Postprocessing functions, required for black-box.
    :type postprocess_fn: Callable[[OVDict], np.ndarray]
    :param explain_mode: Explain mode.
    :type explain_mode: ExplainMode
    :param insertion_parameters: XAI insertion parameters.
    :type insertion_parameters: InsertionParameters]
    """

    def __init__(
        self,
        model: ov.Model,
        task: Task,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        postprocess_fn: Callable[[OVDict], np.ndarray] = None,
        explain_mode: ExplainMode = ExplainMode.AUTO,
        insertion_parameters: InsertionParameters | None = None,
    ) -> None:
        self.model = model
        self.compiled_model: ov.ie_api.CompiledModel | None = None
        self.task = task

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

        self.method = self.create_method(self.explain_mode, self.task)

    def create_method(self, explain_mode: ExplainMode, task: Task) -> MethodBase:
        if explain_mode == ExplainMode.WHITEBOX:
            try:
                method = WhiteBoxMethodFactory.create_method(
                    task, self.model, self.preprocess_fn, self.insertion_parameters
                )
                logger.info("Explaining the model in white-box mode.")
                return method
            except Exception as e:
                print(e)
                raise RuntimeError("Failed to insert XAI into the model. Try to use black-box.")
        elif self.explain_mode == ExplainMode.BLACKBOX:
            self._check_postprocess_fn()
            method = BlackBoxMethodFactory.create_method(task, self.model, self.preprocess_fn, self.postprocess_fn)
            logger.info("Explaining the model in black-box mode.")
            return method
        elif self.explain_mode == ExplainMode.AUTO:
            try:
                method = WhiteBoxMethodFactory.create_method(
                    task, self.model, self.preprocess_fn, self.insertion_parameters
                )
                logger.info("Explaining the model in the white-box mode.")
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model - using black-box mode.")
                self._check_postprocess_fn()
                method = BlackBoxMethodFactory.create_method(task, self.model, self.preprocess_fn, self.postprocess_fn)
                logger.info("Explaining the model in the black-box mode.")
            return method
        else:
            raise ValueError(f"Not supported explain mode {self.explain_mode}.")

    def __call__(
        self,
        data: np.ndarray,
        explanation_parameters: ExplanationParameters,
        **kwargs,
    ) -> Explanation:
        """Explainer call that generates processed explanation result."""
        explain_target_indices = None
        if (
            isinstance(self.method, BlackBoxXAIMethod)
            and explanation_parameters.target_explain_group == TargetExplainGroup.CUSTOM
        ):
            explain_target_indices = get_explain_target_indices(
                explanation_parameters.target_explain_labels,
                explanation_parameters.label_names,
            )

        saliency_map = self.method.generate_saliency_map(
            data,
            explain_target_indices=explain_target_indices,  # type: ignore
            **kwargs,
        )

        explanation = Explanation(
            saliency_map=saliency_map,
            target_explain_group=explanation_parameters.target_explain_group,
            target_explain_labels=explanation_parameters.target_explain_labels,
            label_names=explanation_parameters.label_names,
        )
        return self._visualize(explanation, data, explanation_parameters)

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> OVDict:
        """Forward pass of the compiled model."""
        return self.method.model_forward(x, preprocess)

    def _visualize(
        self, explanation: Explanation, data: np.ndarray, explanation_parameters: ExplanationParameters
    ) -> Explanation:
        if not isinstance(self.preprocess_fn, IdentityPreprocessFN):
            # Assume if preprocess_fn is provided - input data is original image
            explanation = Visualizer(
                explanation=explanation,
                original_input_image=data,
                visualization_parameters=explanation_parameters.visualization_parameters,
            ).run()
        else:
            # preprocess_fn is not provided - assume input data is processed
            explanation = Visualizer(
                explanation=explanation,
                output_size=data.shape[:2],  # resize to model input by default
                visualization_parameters=explanation_parameters.visualization_parameters,
            ).run()
        return explanation

    def _check_postprocess_fn(self) -> None:
        if self.postprocess_fn is None:
            raise ValueError("Postprocess function has to be provided for the black-box mode.")
