# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Interface to insertion Explainable AI (XAI) algorithms into OV IR.
"""


from openvino_xai.insertion.insert_xai_into_model import insert_xai
from openvino_xai.insertion.insert_xai_into_model import insert_xai_into_mapi_wrapper

__all__ = [
    "insert_xai",
    "insert_xai_into_mapi_wrapper",
]
