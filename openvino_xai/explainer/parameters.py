# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import List

from openvino_xai.common.parameters import Method


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
    CUSTOM = "custom"


@dataclass
class VisualizationParameters:
    """
    VisualizationParameters parametrize postprocessing of saliency maps.

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

    scaling: bool = False
    resize: bool = False
    colormap: bool = False
    overlay: bool = False
    overlay_weight: float = 0.5


@dataclass
class ExplanationParameters:
    """
    Parametrize model explanation.

    :parameter target_explain_group: Target explain group.
    :type target_explain_group: TargetExplainGroup
    :param target_explain_labels: List of custom labels to explain, optional. Can be list of integer indices (int),
        or list of names (str) from label_names.
    :type target_explain_labels: List[int | str] | None
    :param label_names: List of all label names.
    :type label_names: List[str] | None
    :parameter visualization_parameters: Post-process parameters.
    :type visualization_parameters: VisualizationParameters
    :param black_box_method: Defines black-box method type.
    :type black_box_method: Method
    """

    target_explain_group: TargetExplainGroup = TargetExplainGroup.CUSTOM
    target_explain_labels: List[int | str] | None = None
    label_names: List[str] | None = None
    visualization_parameters: VisualizationParameters | None = None
    black_box_method: Method = Method.RISE
