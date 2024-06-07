# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Interface for inserting XAI branch into OV IR.
"""
from openvino_xai.inserter.inserter import insert_xai
from openvino_xai.inserter.parameters import (
    ClassificationInsertionParameters,
    DetectionInsertionParameters,
    InsertionParameters,
)

__all__ = [
    "insert_xai",
    "InsertionParameters",
    "ClassificationInsertionParameters",
    "DetectionInsertionParameters",
]
