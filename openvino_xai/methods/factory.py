# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import IdentityPreprocessFN, logger
from openvino_xai.methods.base import MethodBase
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod
from openvino_xai.methods.black_box.rise import RISE
from openvino_xai.methods.white_box.activation_map import ActivationMap
from openvino_xai.methods.white_box.base import WhiteBoxMethod
from openvino_xai.methods.white_box.det_class_probability_map import (
    DetClassProbabilityMap,
)
from openvino_xai.methods.white_box.recipro_cam import ReciproCAM, ViTReciproCAM


class MethodFactory(ABC):
    @classmethod
    @abstractmethod
    def create_method(cls, *args, **kwargs) -> MethodBase:
        """Creates method."""

    @staticmethod
    @abstractmethod
    def create_classification_method(*args, **kwargs) -> MethodBase:
        """Creates classification method."""

    @staticmethod
    @abstractmethod
    def create_detection_method(*args, **kwargs) -> MethodBase:
        """Creates detection method."""


class WhiteBoxMethodFactory(MethodFactory):
    @classmethod
    def create_method(
        cls,
        task: Task,
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        explain_method: Method | None = None,
        target_layer: str | List[str] | None = None,
        embed_scaling: bool | None = True,
        device_name: str = "CPU",
        **kwargs,
    ) -> MethodBase:
        if task == Task.CLASSIFICATION:
            return cls.create_classification_method(
                model,
                preprocess_fn,
                explain_method,
                target_layer,  # type: ignore
                embed_scaling,
                device_name,
                **kwargs,
            )
        if task == Task.DETECTION:
            return cls.create_detection_method(
                model,
                preprocess_fn,
                explain_method,
                target_layer,  # type: ignore
                embed_scaling,
                device_name,
                **kwargs,
            )
        raise ValueError(f"Model type {task} is not supported in white-box mode.")

    @staticmethod
    def create_classification_method(
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        explain_method: Method | None = None,
        target_layer: str | None = None,
        embed_scaling: bool | None = True,
        device_name: str = "CPU",
        **kwargs,
    ) -> WhiteBoxMethod:
        """Generates instance of the classification white-box method class.

        :param model: OV IR model.
        :type model: ov.Model
        :param preprocess_fn: Preprocessing function, identity function by default
            (assume input images are already preprocessed by user).
        :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
        :parameter explain_method: Explain method to use for model explanation.
        :type explain_method: Method
        :parameter target_layer: Target layer(s) (node(s)) name after which the XAI branch will be inserted.
        :type target_layer: str | List[str]
        :parameter embed_scaling: If set to True, saliency map scale (0 ~ 255) operation is embedded in the model.
        :type embed_scaling: bool
        :param device_name: Device type name.
        :type device_name: str
        """

        if target_layer is None:
            logger.info("Target insertion layer is not provided - trying to find it in auto mode.")
        else:
            logger.info(f"Target insertion layer {target_layer} is provided.")

        if explain_method is None or explain_method == Method.RECIPROCAM:
            logger.info("Using ReciproCAM method (for CNNs).")
            return ReciproCAM(
                model,
                preprocess_fn,
                target_layer,
                embed_scaling,
                device_name,
                **kwargs,
            )
        if explain_method == Method.VITRECIPROCAM:
            logger.info("Using ViTReciproCAM method (for vision transformers).")
            return ViTReciproCAM(
                model,
                preprocess_fn,
                target_layer,
                embed_scaling,
                device_name,
                **kwargs,
            )
        if explain_method == Method.ACTIVATIONMAP:
            logger.info("Using ActivationMap method (for CNNs).")
            return ActivationMap(
                model,
                preprocess_fn,
                target_layer,
                embed_scaling,
                device_name,
                **kwargs,
            )
        raise ValueError(f"Requested explanation method {explain_method} is not implemented.")

    @staticmethod
    def create_detection_method(
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        explain_method: Method | None = None,
        target_layer: List[str] | None = None,
        embed_scaling: bool = True,
        device_name: str = "CPU",
        **kwargs,
    ) -> WhiteBoxMethod:
        """Generates instance of the detection white-box method class.

        :param model: OV IR model.
        :type model: ov.Model
        :param preprocess_fn: Preprocessing function, identity function by default
            (assume input images are already preprocessed by user).
        :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
        :parameter explain_method: Explain method to use for model explanation.
        :type explain_method: Method
        :parameter target_layer: Target layer(s) (node(s)) name after which the XAI branch will be inserted.
        :type target_layer: str | List[str]
        :parameter embed_scaling: If set to True, saliency map scale (0 ~ 255) operation is embedded in the model.
        :type embed_scaling: bool
        """

        if target_layer is None:
            raise ValueError("target_layer is required for the detection.")

        if explain_method is None or explain_method == Method.DETCLASSPROBABILITYMAP:
            return DetClassProbabilityMap(
                model=model,
                preprocess_fn=preprocess_fn,
                target_layer=target_layer,
                embed_scaling=embed_scaling,
                device_name=device_name,
                **kwargs,
            )
        raise ValueError(f"Requested explanation method {explain_method} is not implemented.")


class BlackBoxMethodFactory(MethodFactory):
    @classmethod
    def create_method(
        cls,
        task: Task,
        model: ov.Model,
        postprocess_fn: Callable[[OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        **kwargs,
    ) -> MethodBase:
        if task == Task.CLASSIFICATION:
            return cls.create_classification_method(model, postprocess_fn, preprocess_fn, device_name, **kwargs)
        if task == Task.DETECTION:
            return cls.create_detection_method(model, postprocess_fn, preprocess_fn, device_name, **kwargs)
        raise ValueError(f"Model type {task} is not supported in black-box mode.")

    @staticmethod
    def create_classification_method(
        model: ov.Model,
        postprocess_fn: Callable[[OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        **kwargs,
    ) -> BlackBoxXAIMethod:
        """Generates instance of the classification black-box method class.

        :param model: OV IR model.
        :type model: ov.Model
        :param postprocess_fn: Preprocessing function that extract scores from IR model output.
        :type postprocess_fn: Callable[[OVDict], np.ndarray]
        :param preprocess_fn: Preprocessing function, identity function by default
            (assume input images are already preprocessed by user).
        :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
        :param device_name: Device type name.
        :type device_name: str
        """
        return RISE(model, postprocess_fn, preprocess_fn, device_name, **kwargs)

    @staticmethod
    def create_detection_method(*args, **kwargs) -> BlackBoxXAIMethod:
        raise ValueError("Detection models are not supported in black-box mode yet.")
