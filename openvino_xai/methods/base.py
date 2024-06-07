# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable
from venv import logger

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset10 as opset

from openvino_xai.common.utils import SALIENCY_MAP_OUTPUT_NAME, IdentityPreprocessFN
from openvino_xai.inserter.inserter import insert_xai_branch_into_model, has_xai


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

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> ov.utils.data_helpers.wrappers.OVDict:
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


class WhiteBoxMethodBase(MethodBase):
    """
    Base class for white-box XAI methods.

    :param model: OpenVINO model.
    :type model: ov.Model
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param embed_scale: Whether to scale output or not.
    :type embed_scale: bool
    """

    def __init__(
        self,
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        embed_scale: bool = True,
    ):
        super().__init__(preprocess_fn=preprocess_fn)
        self._model_ori = model
        self.preprocess_fn = preprocess_fn
        self.embed_scale = embed_scale

    @property
    def model_ori(self):
        return self._model_ori

    @abstractmethod
    def generate_xai_branch(self):
        """Implements specific XAI algorithm."""

    def generate_saliency_map(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Saliency map generation. White-box implementation."""
        model_output = self.model_forward(data)
        return model_output[SALIENCY_MAP_OUTPUT_NAME]

    def prepare_model(self, load_model: bool = True) -> ov.Model:
        if has_xai(self._model_ori):
            logger.info("Provided IR model already contains XAI branch.")
            self._model = self._model_ori
            if load_model:
                self.load_model()
            return self._model

        xai_output_node = self.generate_xai_branch()
        self._model = insert_xai_branch_into_model(self._model_ori, xai_output_node, self.embed_scale)
        if not has_xai(self._model):
            raise RuntimeError("Insertion of the XAI branch into the model was not successful.")
        if load_model:
            self.load_model()
        return self._model

    @staticmethod
    def _propagate_dynamic_batch_dimension(model: ov.Model):
        # TODO: support models with multiple inputs.
        assert len(model.inputs) == 1, "Support only for models with a single input."
        if not model.input(0).partial_shape[0].is_dynamic:
            partial_shape = model.input(0).partial_shape
            partial_shape[0] = -1  # make batch dimensions to be dynamic
            model.reshape(partial_shape)

    @staticmethod
    def _scale_saliency_maps(saliency_maps: ov.Node, per_class: bool) -> ov.Node:
        """Scale saliency maps to [0, 255] range, per-map."""
        # TODO: unify for per-class and for per-image
        if per_class:
            # Normalization for per-class saliency maps
            _, num_classes, h, w = saliency_maps.get_output_partial_shape(0)
            num_classes, h, w = num_classes.get_length(), h.get_length(), w.get_length()
            saliency_maps = opset.reshape(saliency_maps, (num_classes, h * w), False)
            max_val = opset.unsqueeze(opset.reduce_max(saliency_maps.output(0), [1]), 1)
            min_val = opset.unsqueeze(opset.reduce_min(saliency_maps.output(0), [1]), 1)
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(
                opset.subtract(max_val.output(0), min_val.output(0)), opset.constant(1e-12, dtype=np.float32)
            )
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            saliency_maps = opset.reshape(saliency_maps, (1, num_classes, h, w), False)
            return saliency_maps
        else:
            # Normalization for per-image saliency map
            max_val = opset.reduce_max(saliency_maps.output(0), [0, 1, 2])
            min_val = opset.reduce_min(saliency_maps.output(0), [0, 1, 2])
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(
                opset.subtract(max_val.output(0), min_val.output(0)), opset.constant(1e-12, dtype=np.float32)
            )
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            return saliency_maps


class BlackBoxXAIMethodBase(MethodBase):
    """Base class for methods that explain model in Black-Box mode."""
