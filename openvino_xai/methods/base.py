# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Mapping, TypeVar

import numpy as np
import openvino as ov
import torch

from openvino_xai.common.utils import IdentityPreprocessFN

Model = TypeVar("Model", ov.Model, torch.nn.Module)
CompiledModel = TypeVar("CompiledModel", ov.CompiledModel, torch.nn.Module)


class MethodBase(ABC, Generic[Model, CompiledModel]):
    """Base class for XAI methods."""

    def __new__(
        cls,
        model: Model | None = None,
        *args,
        **kwargs,
    ):
        if isinstance(model, torch.nn.Module):
            raise NotImplementedError(f"{type(model)} is not yet supported for {cls}")
        elif model is not None and not isinstance(model, ov.Model):
            raise ValueError(f"{type(model)} is not supported")
        return super().__new__(cls)

    def __init__(
        self,
        model: Model | None = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
    ):
        self._model = model
        self._model_compiled = None
        self.preprocess_fn = preprocess_fn
        self._device_name = device_name

    @property
    def model_compiled(self) -> CompiledModel | None:
        return self._model_compiled

    @abstractmethod
    def prepare_model(self, load_model: bool = True) -> Model:
        """Model preparation steps."""

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> Mapping:
        """Forward pass of the compiled model. Applies preprocess_fn."""
        if not self._model_compiled:
            raise RuntimeError("Model is not compiled. Call prepare_model() first.")
        if preprocess:
            x = self.preprocess_fn(x)
        return self._model_compiled(x)

    @abstractmethod
    def generate_saliency_map(self, data: np.ndarray) -> Dict[int, np.ndarray] | np.ndarray:
        """Saliency map generation."""


class OVMethod(MethodBase[ov.Model, ov.CompiledModel]):
    def load_model(self) -> None:
        core = ov.Core()
        self._model_compiled = core.compile_model(model=self._model, device_name=self._device_name)


class TorchMethod(MethodBase[torch.nn.Module, torch.nn.Module]):
    pass
