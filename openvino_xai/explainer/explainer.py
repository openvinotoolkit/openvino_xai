# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from openvino_xai import Task
from openvino_xai.common.parameters import Method
from openvino_xai.common.utils import IdentityPreprocessFN, logger
from openvino_xai.explainer.explanation import Explanation
from openvino_xai.explainer.parameters import (
    ExplainMode,
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
        """
        Creates XAI method.

        :param explain_mode: Explain mode.
        :type explain_mode: ExplainMode
        :param task: Type of the task: CLASSIFICATION or DETECTION.
        :type task: Task
        """
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
        target_explain_group: TargetExplainGroup = TargetExplainGroup.CUSTOM,
        target_explain_labels: List[int | str] | None = None,
        label_names: List[str] | None = None,
        scaling: bool = False,
        resize: bool = True,
        colormap: bool = True,
        overlay: bool = False,
        overlay_weight: float = 0.5,
        **kwargs,
    ) -> Explanation:
        return self.explain(
            data,
            target_explain_group,
            target_explain_labels,
            label_names,
            scaling,
            resize,
            colormap,
            overlay,
            overlay_weight,
            **kwargs,
        )

    def explain(
        self,
        data: np.ndarray,
        target_explain_group: TargetExplainGroup = TargetExplainGroup.CUSTOM,
        target_explain_labels: List[int | str] | None = None,
        label_names: List[str] | None = None,
        scaling: bool = False,
        resize: bool = True,
        colormap: bool = True,
        overlay: bool = False,
        overlay_weight: float = 0.5,
        **kwargs,
    ) -> Explanation:
        """
        Interface that generates explanation result.

        :param data: Input image.
        :type data: np.ndarray
        :param target_explain_group: Defines targets to explain: all, only predictions, custom list, per-image.
        :type target_explain_group: TargetExplainGroup
        :param target_explain_labels: List of custom labels to explain, optional. Can be list of integer indices (int),
            or list of names (str) from label_names.
        :type target_explain_labels: List[int | str] | None
        :param label_names: List of all label names.
        :type label_names: List[str] | None
        :parameter scaling: If True, scaling saliency map into [0, 255] range (filling the whole range).
            By default, scaling is embedded into the IR model.
            Therefore, scaling=False here by default.
        :type scaling: bool
        :parameter resize: If True, resize saliency map to the input image size.
        :type resize: bool
        :parameter colormap: If True, apply colormap to the grayscale saliency map.
        :type colormap: bool
        :parameter overlay: If True, generate overlay of the saliency map over the input image.
        :type overlay: bool
        :parameter overlay_weight: Weight of the saliency map when overlaying the input data with the saliency map.
        :type overlay_weight: float
        """
        explain_target_indices = None
        if (
            isinstance(self.method, BlackBoxXAIMethod)
            and target_explain_group == TargetExplainGroup.CUSTOM
        ):
            explain_target_indices = get_explain_target_indices(
                target_explain_labels,
                label_names,
            )

        saliency_map = self.method.generate_saliency_map(
            data,
            explain_target_indices=explain_target_indices,  # type: ignore
            **kwargs,
        )

        explanation = Explanation(
            saliency_map=saliency_map,
            target_explain_group=target_explain_group,
            target_explain_labels=target_explain_labels,
            label_names=label_names,
        )
        return self._visualize(
            explanation,
            data,
            scaling,
            resize,
            colormap,
            overlay,
            overlay_weight,
        )

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> OVDict:
        """Forward pass of the compiled model."""
        return self.method.model_forward(x, preprocess)

    def _visualize(
        self, 
        explanation: Explanation, 
        data: np.ndarray, 
        scaling: bool,
        resize: bool,
        colormap: bool,
        overlay: bool,
        overlay_weight: float,
    ) -> Explanation:
        if not isinstance(self.preprocess_fn, IdentityPreprocessFN):
            # Assume if preprocess_fn is provided - input data is original image
            explanation = Visualizer(
                explanation=explanation,
                original_input_image=data,
                scaling=scaling,
                resize=resize,
                colormap=colormap,
                overlay=overlay,
                overlay_weight=overlay_weight,
            ).run()
        else:
            # preprocess_fn is not provided - assume input data is processed
            explanation = Visualizer(
                explanation=explanation,
                output_size=data.shape[:2],  # resize to model input by default
                scaling=scaling,
                resize=resize,
                colormap=colormap,
                overlay=overlay,
                overlay_weight=overlay_weight,
            ).run()
        return explanation

    def _check_postprocess_fn(self) -> None:
        if self.postprocess_fn is None:
            raise ValueError("Postprocess function has to be provided for the black-box mode.")
