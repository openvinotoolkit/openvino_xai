# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Interface for getting explanation.
"""
from openvino_xai.explanation.explanation_parameters import ExplanationParameters, ExplainMode, TargetExplainGroup, \
    PostProcessParameters

from openvino_xai.explanation.explain_model import explain

__all__ = [
    "explain",
    "ExplanationParameters",
    "ExplainMode",
    "TargetExplainGroup",
    "PostProcessParameters",
]
