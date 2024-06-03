# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO-XAI library for explaining OpenVINO™ IR models.
"""


from .common.parameters import Method, Task
from .xai_branch_inserter import insert_xai

__all__ = ["insert_xai", "Method", "Task"]
