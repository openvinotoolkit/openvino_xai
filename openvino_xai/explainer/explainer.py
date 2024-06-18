# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Callable, List

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from openvino_xai import Task
from openvino_xai.common.parameters import Method
from openvino_xai.common.utils import IdentityPreprocessFN, logger
from openvino_xai.explainer.explanation import Explanation
from openvino_xai.explainer.utils import explain_all, get_explain_target_indices
from openvino_xai.explainer.visualizer import Visualizer
from openvino_xai.methods.base import MethodBase
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod
from openvino_xai.methods.factory import BlackBoxMethodFactory, WhiteBoxMethodFactory


class ExplainMode(Enum):
    """
    Enum describes different explain modes.

    Contains the following values:
        WHITEBOX - The model is explained in white box mode, i.e. XAI branch is getting inserted into the model graph.
        BLACKBOX - The model is explained in black box model.
    """

    WHITEBOX = "whitebox"
    BLACKBOX = "blackbox"
    AUTO = "auto"


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
    :parameter explain_method: Explain method to use for model explanation.
    :type explain_method: Method
    :parameter target_layer: Target layer(s) (node(s)) name after which the XAI branch will be inserted.
    :type target_layer: str | List[str]
    :parameter embed_scaling: If set to True, saliency map scale (0 ~ 255) operation is embedded in the model.
    :type embed_scaling: bool
    """

    def __init__(
        self,
        model: ov.Model,
        task: Task,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        postprocess_fn: Callable[[OVDict], np.ndarray] = None,
        explain_mode: ExplainMode = ExplainMode.AUTO,
        explain_method: Method | None = None,
        target_layer: str | List[str] | None = None,
        embed_scaling: bool | None = True,
        **kwargs,
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

        self.target_layer = target_layer
        self.embed_scaling = embed_scaling
        self.explain_method = explain_method
        self.white_box_method_kwargs = kwargs

        self.explain_mode = explain_mode

        self.visualizer = Visualizer()

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
                return self._create_white_box_method(task)
            except Exception as e:
                print(e)
                raise RuntimeError("Failed to insert XAI into the model. Try to use black-box.")
        elif self.explain_mode == ExplainMode.BLACKBOX:
            return self._create_black_box_method(task)
        elif self.explain_mode == ExplainMode.AUTO:
            try:
                return self._create_white_box_method(task)
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model - using black-box mode.")
                return self._create_black_box_method(task)
        else:
            raise ValueError(f"Not supported explain mode {self.explain_mode}.")

    def __call__(
        self,
        data: np.ndarray,
        targets: List[int | str] | int | str,
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
            targets,
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
        targets: List[int | str] | int | str,
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
        :param targets: List of custom labels to explain, optional. Can be list of integer indices (int),
            or list of names (str) from label_names.
        :type targets: List[int | str] | int | str
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
        if isinstance(targets, (int, str)):
            targets = [targets]

        explain_target_indices = None
        if isinstance(self.method, BlackBoxXAIMethod) and not explain_all(targets):
            explain_target_indices = get_explain_target_indices(
                targets,
                label_names,
            )

        saliency_map = self.method.generate_saliency_map(
            data,
            explain_target_indices=explain_target_indices,  # type: ignore
            **kwargs,
        )

        explanation = Explanation(
            saliency_map=saliency_map,
            targets=targets,
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

    def _create_white_box_method(self, task: Task) -> MethodBase:
        method = WhiteBoxMethodFactory.create_method(
            task,
            self.model,
            self.preprocess_fn,
            self.explain_method,
            self.target_layer,
            self.embed_scaling,
            **self.white_box_method_kwargs,
        )
        logger.info("Explaining the model in white-box mode.")
        return method

    def _create_black_box_method(self, task: Task) -> MethodBase:
        if self.postprocess_fn is None:
            raise ValueError("Postprocess function has to be provided for the black-box mode.")
        method = BlackBoxMethodFactory.create_method(task, self.model, self.preprocess_fn, self.postprocess_fn)
        logger.info("Explaining the model in black-box mode.")
        return method

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
            explanation = self.visualizer(
                explanation=explanation,
                original_input_image=data,
                scaling=scaling,
                resize=resize,
                colormap=colormap,
                overlay=overlay,
                overlay_weight=overlay_weight,
            )
        else:
            explanation = self.visualizer(
                explanation=explanation,
                output_size=data.shape[:2],  # resize to model input by default
                scaling=scaling,
                resize=resize,
                colormap=colormap,
                overlay=overlay,
                overlay_weight=overlay_weight,
            )
        return explanation
