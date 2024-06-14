# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


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
