# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO-XAI library for explaining OpenVINO™ IR models.
"""


from .insertion import insert_xai
from .explanation import explain

__all__ = [
    "insert_xai",
    "explain",
]
