# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ModelType(Enum):
    """Enum representing the different model types."""

    CNN = "cnn"
    TRANSFORMER = "transformer"
