# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset10 as opset

from openvino_xai.common.utils import IdentityPreprocessFN
from openvino_xai.inserter.model_parser import IRParserCls, ModelType
from openvino_xai.methods.white_box.base import WhiteBoxMethod


class ActivationMap(WhiteBoxMethod):
    """
    Implements ActivationMap.

    :param model: OpenVINO model.
    :type model: ov.Model
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :type target_layer: str
    :param embed_scaling: Whether to scale output or not.
    :type embed_scaling: bool
    :param device_name: Device type name.
    :type device_name: str
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        target_layer: str | None = None,
        embed_scaling: bool = True,
        device_name: str = "CPU",
        prepare_model: bool = True,
    ):
        super().__init__(model=model, preprocess_fn=preprocess_fn, embed_scaling=embed_scaling, device_name=device_name)
        self.per_class = False
        self.model_type = ModelType.CNN
        self._target_layer = target_layer

        if prepare_model:
            self.prepare_model()

    def generate_xai_branch(self) -> ov.Node:
        """Implements ActivationMap XAI algorithm."""
        target_node_ori = IRParserCls.get_target_node(self._model_ori, self.model_type, self._target_layer)
        saliency_maps = opset.reduce_mean(target_node_ori.output(0), 1)
        if self.embed_scaling:
            saliency_maps = self._scale_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps
