# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Interface for inserting XAI branch into OV IR.
"""
from openvino_xai.xai_branch_inserter.insertion_parameters import (
    ClassificationInsertionParameters,
    DetectionInsertionParameters,
    InsertionParameters,
)
from openvino_xai.xai_branch_inserter.xai_branch_inserter import insert_xai

__all__ = [
    "insert_xai",
    "InsertionParameters",
    "ClassificationInsertionParameters",
    "DetectionInsertionParameters",
]
