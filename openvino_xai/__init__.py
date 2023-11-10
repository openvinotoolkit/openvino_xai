# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Openvino-XAI library for explaining OpenVINO™ IR models.
"""


from openvino_xai.insertion import insert_xai
from openvino_xai.explanation import explain

__all__ = [
    "insert_xai",
    "explain",
]
