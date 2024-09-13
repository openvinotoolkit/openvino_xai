# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import abstractmethod
from typing import Callable

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset10 as opset

from openvino_xai.common.utils import (
    SALIENCY_MAP_OUTPUT_NAME,
    IdentityPreprocessFN,
    has_xai,
    logger,
)
from openvino_xai.inserter.inserter import insert_xai_branch_into_model
from openvino_xai.methods.base import MethodBase


class WhiteBoxMethod(MethodBase[ov.Model, ov.CompiledModel]):
    """
    Base class for white-box XAI methods.

    :param model: OpenVINO model.
    :type model: ov.Model
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param embed_scaling: Whether to scale output or not.
    :type embed_scaling: bool
    :param device_name: Device type name.
    :type device_name: str
    """

    def __init__(
        self,
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        embed_scaling: bool = True,
        device_name: str = "CPU",
    ):
        super().__init__(preprocess_fn=preprocess_fn, device_name=device_name)
        self._model_ori = copy.deepcopy(model)
        self.embed_scaling = embed_scaling

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
        if self._model is None:
            if has_xai(self._model_ori):
                logger.info("Provided IR model already contains XAI branch.")
                self._model = self._model_ori
            else:
                xai_output_node = self.generate_xai_branch()
                self._model = insert_xai_branch_into_model(self._model_ori, xai_output_node, self.embed_scaling)
                if not has_xai(self._model):
                    raise RuntimeError("Insertion of the XAI branch into the model was not successful.")
        if load_model and self._model_compiled is None:
            self._model_compiled = ov.Core().compile_model(model=self._model, device_name=self._device_name)
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
            # Scale for per-class saliency maps
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
            # Scale for per-image saliency map
            max_val = opset.reduce_max(saliency_maps.output(0), [0, 1, 2])
            min_val = opset.reduce_min(saliency_maps.output(0), [0, 1, 2])
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(
                opset.subtract(max_val.output(0), min_val.output(0)), opset.constant(1e-12, dtype=np.float32)
            )
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            return saliency_maps
