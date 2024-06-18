# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


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
