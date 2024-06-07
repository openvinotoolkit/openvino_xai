# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

import openvino.runtime as ov
from openvino_xai.common.parameters import Task
from openvino_xai.common.utils import IdentityPreprocessFN, logger
from openvino_xai.inserter.parameters import ClassificationInsertionParameters, DetectionInsertionParameters
from openvino_xai.common.parameters import Method
from openvino_xai.methods.black_box.black_box_methods import RISE, BlackBoxXAIMethodBase
from openvino_xai.methods.white_box.white_box_methods import ActivationMapXAIMethod, DetClassProbabilityMapXAIMethod, ReciproCAMXAIMethod, ViTReciproCAMXAIMethod, WhiteBoxXAIMethodBase


class MethodFactory(ABC):
    @classmethod
    @abstractmethod
    def create_method(cls):
        """Creates method."""

    @staticmethod
    @abstractmethod
    def create_classification_method():
        """Creates classification method."""

    @staticmethod
    @abstractmethod
    def create_detection_method():
        """Creates detection method."""


class WhiteBoxMethodFactory(MethodFactory):
    @classmethod
    def create_method(
        cls,
        task: Task,
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        insertion_parameters: ClassificationInsertionParameters | None = None,
        **kwargs,
    ) -> WhiteBoxXAIMethodBase:
        if task == Task.CLASSIFICATION:
            return cls.create_classification_method(model, preprocess_fn, insertion_parameters, **kwargs)
        if task == Task.DETECTION:
            return cls.create_detection_method(model, preprocess_fn, insertion_parameters, **kwargs)
        raise ValueError(f"Model type {task} is not supported in white-box mode.")

    @staticmethod
    def create_classification_method(
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        insertion_parameters: ClassificationInsertionParameters | None = None,
        **kwargs,
    ) -> WhiteBoxXAIMethodBase:
        """Generates instance of the classification white-box method class.

        :param model: OV IR model.
        :type model: ov.Model
        :param preprocess_fn: Preprocessing function, identity function by default
            (assume input images are already preprocessed by user).
        :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
        :param insertion_parameters: Insertion parameters that parametrize explanation method,
            that will be inserted into the model graph.
        :type insertion_parameters: InsertionParameters
        """

        if insertion_parameters is None or insertion_parameters.target_layer is None:
            logger.info("Target insertion layer is not provided - trying to find it in auto mode.")
        else:
            logger.info(f"Target insertion layer {insertion_parameters.target_layer} is provided.")

        if insertion_parameters is None:
            logger.info("Using ReciproCAM method (for CNNs).")
            return ReciproCAMXAIMethod(model, preprocess_fn, **kwargs)

        explain_method = insertion_parameters.explain_method
        if explain_method == Method.RECIPROCAM:
            logger.info("Using ReciproCAM method (for CNNs).")
            return ReciproCAMXAIMethod(
                model,
                preprocess_fn,
                insertion_parameters.target_layer,
                insertion_parameters.embed_normalization,
                **kwargs,
            )
        if explain_method == Method.VITRECIPROCAM:
            logger.info("Using ViTReciproCAM method (for vision transformers).")
            return ViTReciproCAMXAIMethod(
                model,
                preprocess_fn,
                insertion_parameters.target_layer,
                insertion_parameters.embed_normalization,
                **kwargs,
            )
        if explain_method == Method.ACTIVATIONMAP:
            logger.info("Using ActivationMap method (for CNNs).")
            return ActivationMapXAIMethod(
                model,
                preprocess_fn,
                insertion_parameters.target_layer, 
                insertion_parameters.embed_normalization,
                **kwargs,
            )
        raise ValueError(f"Requested explanation method {explain_method} is not implemented.")

    @staticmethod
    def create_detection_method(
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        insertion_parameters: DetectionInsertionParameters,
        **kwargs,
    ) -> WhiteBoxXAIMethodBase:
        """Generates instance of the detection white-box method class.

        :param model: OV IR model.
        :type model: ov.Model
        :param preprocess_fn: Preprocessing function, identity function by default
            (assume input images are already preprocessed by user).
        :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
        :param insertion_parameters: Insertion parameters that parametrize explanation method,
            that will be inserted into the model graph.
        :type insertion_parameters: InsertionParameters
        """

        if insertion_parameters is None:
            raise ValueError("insertion_parameters is required for the detection models.")

        explain_method = insertion_parameters.explain_method
        if explain_method == Method.DETCLASSPROBABILITYMAP:
            return DetClassProbabilityMapXAIMethod(
                model,
                insertion_parameters.target_layer,
                preprocess_fn,
                insertion_parameters.num_anchors,
                insertion_parameters.saliency_map_size,
                insertion_parameters.embed_normalization,
                **kwargs,
            )
        raise ValueError(f"Requested explanation method {explain_method} is not implemented.")


class BlackBoxMethodFactory(MethodFactory):
    @classmethod
    def create_method(
        cls,
        task,
        model,
        preprocess_fn,
        postprocess_fn,
        **kwargs,
    ) -> BlackBoxXAIMethodBase:
        if task == Task.CLASSIFICATION:
            return cls.create_classification_method(model, postprocess_fn, preprocess_fn, **kwargs)
        if task == Task.DETECTION:
            return cls.create_detection_method(model, preprocess_fn, preprocess_fn, **kwargs)
        raise ValueError(f"Model type {task} is not supported in black-box mode.")

    @staticmethod
    def create_classification_method(
        model: ov.Model,
        postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        **kwargs,
    ) -> BlackBoxXAIMethodBase:
        """Generates instance of the classification black-box method class.

        :param model: OV IR model.
        :type model: ov.Model
        :param postprocess_fn: Preprocessing function that extract scores from IR model output.
        :type postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray]
        :param preprocess_fn: Preprocessing function, identity function by default
            (assume input images are already preprocessed by user).
        :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
        """
        return RISE(model, postprocess_fn, preprocess_fn, **kwargs)

    @staticmethod
    def create_detection_method(*args, **kwargs) -> BlackBoxXAIMethodBase:
        raise ValueError(f"Detection models are not supported in black-box mode yet.")
