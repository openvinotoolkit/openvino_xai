# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import openvino.runtime as ov

from openvino_xai.methods.base import MethodBase


class BlackBoxXAIMethod(MethodBase):
    """Base class for methods that explain model in Black-Box mode."""

    def prepare_model(self, load_model: bool = True) -> ov.Model:
        if load_model:
            self.load_model()
        return self._model

    def get_num_classes(self, data_preprocessed):
        forward_output = self.model_forward(data_preprocessed, preprocess=False)
        logits = self.postprocess_fn(forward_output)
        _, num_classes = logits.shape
        return num_classes


class Preset(Enum):
    """
    Enum representing the different presets:

    Contains the following values:
        SPEED - Prioritizes getting results faster.
        BALANCE - Balance between speed and quality.
        QUALITY - Prioritizes getting high quality results.
    """

    SPEED = "speed"
    BALANCE = "balance"
    QUALITY = "quality"
