# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Callable, Mapping

import numpy as np
import openvino.runtime as ov

from openvino_xai.common.utils import IdentityPreprocessFN
from openvino_xai.methods.base import MethodBase
from openvino_xai.methods.black_box.utils import check_classification_output


class BlackBoxXAIMethod(MethodBase[ov.Model, ov.CompiledModel]):
    """Base class for methods that explain model in Black-Box mode."""

    def __init__(
        self,
        model: ov.Model,
        postprocess_fn: Callable[[Mapping], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
    ):
        super().__init__(model=model, preprocess_fn=preprocess_fn, device_name=device_name)
        self.postprocess_fn = postprocess_fn

    def prepare_model(self, load_model: bool = True) -> ov.Model:
        """Load model prior to inference."""
        if load_model:
            self._model_compiled = ov.Core().compile_model(model=self._model, device_name=self._device_name)
        return self._model

    def get_logits(self, data_preprocessed: np.ndarray) -> np.ndarray:
        """Gets logits for the classification model. Expects batch dimention."""
        forward_output = self.model_forward(data_preprocessed, preprocess=False)
        logits = self.postprocess_fn(forward_output)
        check_classification_output(logits)
        return logits


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
