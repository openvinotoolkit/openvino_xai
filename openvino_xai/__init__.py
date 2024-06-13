# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO-XAI library for explaining OpenVINO™ IR models.
"""


from .api.api import insert_xai
from .common.parameters import Method, Task
from .explainer.explainer import Explainer

__all__ = ["Explainer", "insert_xai", "Method", "Task"]
