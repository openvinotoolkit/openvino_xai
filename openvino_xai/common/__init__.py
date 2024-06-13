# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Common parameters and utils.
"""
from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import has_xai, scaling

__all__ = [
    "Task",
    "Method",
    "has_xai",
    "scaling",
]
