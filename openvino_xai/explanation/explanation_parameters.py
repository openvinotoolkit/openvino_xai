# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict

from openvino_xai.common.parameters import XAIMethodType


class ExplainMode(Enum):
    """
    Enum describes different explain modes.

    Contains the following values:
        WHITEBOX - The model is explained in white box mode, i.e. XAI branch is getting inserted into the model graph.
        BLACKBOX - The model is explained in black box model.
    """

    WHITEBOX = "whitebox"
    BLACKBOX = "blackbox"


class TargetExplainGroup(Enum):
    """
    Enum describes different target explanation groups.

    Contains the following values:
        IMAGE - Global (single) saliency map per image.
        ALL - Saliency map per each possible target.
        PREDICTIONS - Saliency map per each prediction (prediction = target).
        CUSTOM - Saliency map per each specified target.
    """

    IMAGE = "image"
    ALL = "all"
    PREDICTIONS = "predictions"
    CUSTOM = "custom"


SELECTED_TARGETS = {
    TargetExplainGroup.PREDICTIONS,
    TargetExplainGroup.CUSTOM,
}


@dataclass
class PostProcessParameters:
    """
    PostProcessParameters parametrize postprocessing of saliency maps.

    :parameter normalize: If True, normalize saliency map into [0, 255] range (filling the whole range).
        By default, normalization to [0, 255] range is embedded into the IR model.
        Therefore, normalize=False here by default.
    :type normalize: bool
    :parameter resize: If True, resize saliency map to the input image size.
    :type resize: bool
    :parameter colormap: If True, apply colormap to the grayscale saliency map.
    :type colormap: bool
    :parameter overlay: If True, generate overlay of the saliency map over the input image.
    :type overlay: bool
    :parameter overlay_weight: Weight of the saliency map when overlaying the input data with the saliency map.
    :type overlay_weight: float
    """

    normalize: bool = False
    resize: bool = False
    colormap: bool = False
    overlay: bool = False
    overlay_weight: float = 0.5


@dataclass
class ExplanationParameters:
    """
    Parametrize model explanation.

    :parameter explain_mode: Explanation mode.
    :type explain_mode: ExplainMode
    :parameter target_explain_group: Target explain group.
    :type target_explain_group: TargetExplainGroup
    :param custom_target_indices: List of indices of custom targets, optional.
    :type custom_target_indices: Optional[List[int]]
    :parameter post_processing_parameters: Post-process parameters.
    :type post_processing_parameters: PostProcessParameters
    :parameter confidence_threshold: Prediction confidence threshold.
    :type confidence_threshold: float
    :param black_box_method: Defines black-box method type.
    :type black_box_method: XAIMethodType
    :param black_box_method_kwargs: Defines black-box method kwargs.
    :type black_box_method_kwargs: dict
    """

    explain_mode: ExplainMode = ExplainMode.WHITEBOX
    target_explain_group: TargetExplainGroup = TargetExplainGroup.PREDICTIONS
    explain_target_names: Optional[List[str]] = None
    custom_target_indices: Optional[List[int]] = None
    post_processing_parameters: PostProcessParameters = PostProcessParameters(overlay=True)
    confidence_threshold: float = 0.5
    black_box_method: XAIMethodType = XAIMethodType.RISE
    black_box_method_kwargs: Dict = field(default_factory=dict)


class SaliencyMapLayout(Enum):
    """
    Enum describes different saliency map layouts.

    Saliency map can have the following layout:
        ONE_MAP_PER_IMAGE_GRAY - BHW - one map per image
        ONE_MAP_PER_IMAGE_COLOR - BHWC - one map per image, colormapped
        MULTIPLE_MAPS_PER_IMAGE_GRAY - BNHW - multiple maps per image
        MULTIPLE_MAPS_PER_IMAGE_COLOR - BNHWC - multiple maps per image, colormapped
    """

    ONE_MAP_PER_IMAGE_GRAY = "one_map_per_image_gray"
    ONE_MAP_PER_IMAGE_COLOR = "one_map_per_image_color"
    MULTIPLE_MAPS_PER_IMAGE_GRAY = "MULTIPLE_MAPS_PER_IMAGE_GRAY"
    MULTIPLE_MAPS_PER_IMAGE_COLOR = "MULTIPLE_MAPS_PER_IMAGE_COLOR"


GRAY_LAYOUTS = {
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY,
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
}
COLOR_MAPPED_LAYOUTS = {
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR,
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
MULTIPLE_MAP_LAYOUTS = {
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
ONE_MAP_LAYOUTS = {
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY,
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR,
}
