# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TaskType(Enum):
    """
    Enum representing the different task types:

    Contains the following values:
        CLASSIFICATION - Classification task.
        DETECTION - Detection task.
    """

    CLASSIFICATION = "Classification"
    DETECTION = "Detection"


class XAIMethodType(Enum):
    """
    Enum representing the different XAI methods:

    Contains the following values:
        ACTIVATIONMAP - ActivationMap method.
        RECIPROCAM - ReciproCAM method.
        VITRECIPROCAM - VITReciproCAM method.
        DETCLASSPROBABILITYMAP - DetClassProbabilityMap method.
        RISE - RISE method.
    """

    ACTIVATIONMAP = "activationmap"
    RECIPROCAM = "reciprocam"
    VITRECIPROCAM = "vitreciprocam"
    DETCLASSPROBABILITYMAP = "detclassprobabilitymap"
    RISE = "RISE"


WhiteBoxXAIMethods = {
    XAIMethodType.ACTIVATIONMAP,
    XAIMethodType.RECIPROCAM,
    XAIMethodType.DETCLASSPROBABILITYMAP,
}
BlackBoxXAIMethods = {
    XAIMethodType.RISE,
}
ClassificationXAIMethods = {
    XAIMethodType.ACTIVATIONMAP,
    XAIMethodType.RECIPROCAM,
    XAIMethodType.RISE,
}
DetectionXAIMethods = {
    XAIMethodType.DETCLASSPROBABILITYMAP,
}
