# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO-XAI library for explaining OpenVINO™ IR models.
"""


from .insertion import insert_xai

__all__ = [
    "insert_xai",
]


##------------------------
#
# from enum import Enum
# from abc import ABC
# from abc import abstractmethod
# import numpy as np
# import openvino.runtime as ov
#
#
# class XAI:
#
#    class Task(Enum):
#        CLASSIFICATION = "Classification"
#        DETECTION = "Detection"
#
#    class Mode(Enum):
#        WHITEBOX = "whitebox"
#        BLACKBOX = "blackbox"
#        AUTO = "auto"
#
#    class Target(Enum):
#        IMAGE = "image"
#        ALL = "all"
#        CUSTOM = "custom"
#
#    class Algorithm(ABC):
#        @abstractmethod
#        def run(self, image: np.ndarray):
#
#    def __init__(
#        self,
#        model: ov.Model,
#    ):
#        pass
#
#    def explain(
#        self,
#        image: np.ndarray,
#    ):
#        pass
#
#
#
#
##-------------------------
#
# from openvino_xai import XAI
##from openvino_xai.prepostproc import PreProcessor
##from openvino_xai.prepostproc import PostProcessor
#
#
# model = ov.Model()
# image = np.ndarray()
#
# xai = XAI(model)
# result = xai.explain(image)
# result = xai(image)
#
# explainer = Explainer(model)
# result = explainer(image)
