# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Common parameters and utils.
"""
from openvino_xai.common.parameters import TaskType, XAIMethodType
from openvino_xai.common.utils import has_xai, normalize

__all__ = [
    "TaskType",
    "XAIMethodType",
    "has_xai",
    "normalize",
]
