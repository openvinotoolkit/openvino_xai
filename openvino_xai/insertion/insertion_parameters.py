# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from openvino_xai.common.parameters import XAIMethodType


@dataclass
class InsertionParameters:
    """Parametrize explanation method that will be inserted into the model graph
    Applicable for white-box methods."""


@dataclass
class ClassificationInsertionParameters(InsertionParameters):
    """
    Insertion parameters for classification models.

    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :type target_layer: str
    :parameter embed_normalization: If set to True, saliency map normalization is embedded in the model.
    :type embed_normalization: bool
    :parameter explain_method_type: Explain method to use for model explanation.
    :type explain_method_type: openvino_xai.common.parameters.XAIMethodType
    """

    target_layer: str | None = None
    embed_normalization: bool | None = True

    explain_method_type: XAIMethodType = XAIMethodType.RECIPROCAM


@dataclass
class DetectionInsertionParameters(InsertionParameters):
    """
    Insertion parameters for detection models.

    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :type target_layer: str
    :parameter num_anchors: Number of anchors per scale.
    :type num_anchors: List[int]
    :parameter saliency_map_size: Size of the output saliency map.
    :type saliency_map_size: Tuple[int, int] | List[int]
    :parameter embed_normalization: If set to True, saliency map normalization is embedded in the model.
    :type embed_normalization: bool
    :parameter explain_method_type: Explain method to use for model explanation.
    :type explain_method_type: XAIMethodType
    """

    target_layer: List[str]
    num_anchors: List[int] | None = None
    saliency_map_size: Tuple[int, int] | List[int] = (23, 23)
    embed_normalization: bool = True

    explain_method_type: XAIMethodType = XAIMethodType.DETCLASSPROBABILITYMAP


class ModelType(Enum):
    """Enum representing the different model types."""

    CNN = "cnn"
    TRANSFORMER = "transformer"
