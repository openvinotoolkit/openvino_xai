# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from openvino_xai.common.utils import IdentityPreprocessFN


class MethodBase(ABC):
    """Base class for XAI methods."""

    def __init__(
        self,
        model: ov.Model = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
    ):
        self._model = model
        self._model_compiled = None
        self.preprocess_fn = preprocess_fn

    @property
    def model_compiled(self) -> ov.ie_api.CompiledModel | None:
        return self._model_compiled

    @abstractmethod
    def prepare_model(self) -> ov.Model:
        """Model preparation steps."""

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> OVDict:
        """Forward pass of the compiled model. Applies preprocess_fn."""
        if not self._model_compiled:
            raise RuntimeError("Model is not compiled. Call prepare_model() first.")
        if preprocess:
            x = self.preprocess_fn(x)
        return self._model_compiled(x)

    @abstractmethod
    def generate_saliency_map(self, data: np.ndarray) -> np.ndarray:
        """Saliency map generation."""

    def load_model(self) -> None:
        # TODO: support other devices?
        self._model_compiled = ov.Core().compile_model(self._model, "CPU")
