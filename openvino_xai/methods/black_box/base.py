# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import openvino.runtime as ov

from openvino_xai.methods.base import OVMethod
from openvino_xai.methods.black_box.utils import check_classification_output


class BlackBoxXAIMethod(OVMethod):
    """Base class for methods that explain model in Black-Box mode."""

    def prepare_model(self, load_model: bool = True) -> ov.Model:
        """Load model prior to inference."""
        if load_model:
            self.load_model()
        return self._model

    def get_num_classes(self, data_preprocessed):
        """Estimates number of classes for the classification model. Expects batch dimention."""
        forward_output = self.model_forward(data_preprocessed, preprocess=False)
        logits = self.postprocess_fn(forward_output)
        check_classification_output(logits)
        return logits.shape[1]


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
