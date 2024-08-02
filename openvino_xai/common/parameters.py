# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class Task(Enum):
    """
    Enum representing the different task types:

    Contains the following values:
        CLASSIFICATION - Classification task.
        DETECTION - Detection task.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"


class Method(Enum):
    """
    Enum representing the different XAI methods:

    Contains the following values:
        ACTIVATIONMAP - ActivationMap method.
        RECIPROCAM - ReciproCAM method.
        VITRECIPROCAM - VITReciproCAM method.
        DETCLASSPROBABILITYMAP - DetClassProbabilityMap method.
        RISE - RISE method.
        AISE - AISE method.
    """

    ACTIVATIONMAP = "activationmap"
    RECIPROCAM = "reciprocam"
    VITRECIPROCAM = "vitreciprocam"
    DETCLASSPROBABILITYMAP = "detclassprobabilitymap"
    RISE = "rise"
    AISE = "aise"


WhiteBoxXAIMethods = {
    Method.ACTIVATIONMAP,
    Method.RECIPROCAM,
    Method.DETCLASSPROBABILITYMAP,
}
BlackBoxXAIMethods = {
    Method.RISE,
    Method.AISE,
}
ClassificationXAIMethods = {
    Method.ACTIVATIONMAP,
    Method.RECIPROCAM,
    Method.RISE,
    Method.AISE,
}
DetectionXAIMethods = {
    Method.DETCLASSPROBABILITYMAP,
}
