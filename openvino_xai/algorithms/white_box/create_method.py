# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from openvino import runtime as ov

from openvino_xai.algorithms.white_box.white_box_methods import (
    WhiteBoxXAIMethodBase,
    ReciproCAMXAIMethod,
    ActivationMapXAIMethod,
    DetClassProbabilityMapXAIMethod,
    ViTReciproCAMXAIMethod,
)
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters, DetectionInsertionParameters
from openvino_xai.common.parameters import XAIMethodType
from openvino_xai.common.utils import logger


def create_white_box_classification_explain_method(
    model: ov.Model, insertion_parameters: Optional[ClassificationInsertionParameters] = None
) -> WhiteBoxXAIMethodBase:
    """Generates instance of the classification explanation method class.

    :param model: OV IR model.
    :type model: openvino.runtime.Model
    :param insertion_parameters: Explain parameters that parametrize explanation method,
        that will be inserted into the model graph.
    :type insertion_parameters: InsertionParameters
    """

    if insertion_parameters is None or insertion_parameters.target_layer is None:
        logger.info("Target insertion layer is not provided - trying to find it in auto mode.")
    else:
        logger.info(f"Target insertion layer {insertion_parameters.target_layer} is provided.")

    if insertion_parameters is None:
        logger.info("Using ReciproCAM method (for CNNs).")
        return ReciproCAMXAIMethod(model)

    explain_method_type = insertion_parameters.explain_method_type
    if explain_method_type == XAIMethodType.RECIPROCAM:
        logger.info("Using ReciproCAM method (for CNNs).")
        return ReciproCAMXAIMethod(
            model,
            insertion_parameters.target_layer,
            insertion_parameters.embed_normalization,
        )
    if explain_method_type == XAIMethodType.VITRECIPROCAM:
        logger.info("Using ViTReciproCAM method (for vision transformers).")
        return ViTReciproCAMXAIMethod(
            model,
            insertion_parameters.target_layer,
            insertion_parameters.embed_normalization,
            **insertion_parameters.white_box_method_kwargs,
        )
    if explain_method_type == XAIMethodType.ACTIVATIONMAP:
        logger.info("Using ActivationMap method (for CNNs).")
        return ActivationMapXAIMethod(
            model, insertion_parameters.target_layer, insertion_parameters.embed_normalization
        )
    raise ValueError(f"Requested explanation method {explain_method_type} is not implemented.")


def create_white_box_detection_explain_method(
    model: ov.Model, insertion_parameters: DetectionInsertionParameters
) -> WhiteBoxXAIMethodBase:
    """Generates instance of the detection explanation method class.

    :param model: OV IR model.
    :type model: openvino.runtime.Model
    :param insertion_parameters: Explain parameters that parametrize explanation method,
        that will be inserted into the model graph.
    :type insertion_parameters: InsertionParameters
    """

    if insertion_parameters is None:
        raise ValueError("insertion_parameters is required for the detection models.")

    explain_method_type = insertion_parameters.explain_method_type
    if explain_method_type == XAIMethodType.DETCLASSPROBABILITYMAP:
        return DetClassProbabilityMapXAIMethod(
            model,
            insertion_parameters.target_layer,
            insertion_parameters.num_anchors,
            insertion_parameters.saliency_map_size,
            insertion_parameters.embed_normalization,
        )
    raise ValueError(f"Requested explanation method {explain_method_type} is not implemented.")
