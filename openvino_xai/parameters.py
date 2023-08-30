# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

from openvino_xai.methods import XAIMethodType


@dataclass
class ExplainParameters:
    """ExplainParameters parametrize explain method that will be inserted into the model graph
    Applicable for white-box methods."""


@dataclass
class ClassificationExplainParametersWB(ExplainParameters):
    """
    Explain parameters for classification.

    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :parameter embed_normalization: If set to True, saliency map normalization is embedded in the model.
    :parameter explain_method_type: Explain method to use for model explanation.
    """

    target_layer: Optional[str] = None
    embed_normalization: Optional[bool] = True

    explain_method_type: XAIMethodType = XAIMethodType.RECIPROCAM


@dataclass
class DetectionExplainParametersWB(ExplainParameters):
    """
    Explain parameters for detection.

    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :parameter num_anchors: Number of anchors per scale.
    :parameter saliency_map_size: Size of the output saliency map.
    :parameter embed_normalization: If set to True, saliency map normalization is embedded in the model.
    :parameter explain_method_type: Explain method to use for model explanation.
    """

    target_layer: List[str]
    num_anchors: List[int]
    saliency_map_size: Union[Tuple[int, int], List[int]] = (13, 13)
    embed_normalization: bool = True

    explain_method_type: XAIMethodType = XAIMethodType.DETCLASSPROBABILITYMAP


@dataclass
class PostProcessParameters:
    """
    PostProcessParameters parametrize postprocessing of saliency maps.

    :parameter normalize: If True, normalize saliency map into [0, 255] range (filling the whole range).
        By default, normalization to [0, 255] range is embedded into the IR model.
        Therefore, normalize=False here by default.
    :parameter resize: If True, resize saliency map to the input image size.
    :parameter colormap: If True, apply colormap to the grayscale saliency map.
    :parameter overlay: If True, generate overlay of the saliency map over the input image.
    :parameter overlay_weight: Weight of the saliency map when overlaying the input data with the saliency map.
    """

    normalize: bool = False
    resize: bool = False
    colormap: bool = False
    overlay: bool = False
    overlay_weight: float = 0.5
