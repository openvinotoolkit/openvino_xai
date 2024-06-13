# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_xai.methods.base import MethodBase


class BlackBoxXAIMethod(MethodBase):
    """Base class for methods that explain model in Black-Box mode."""
