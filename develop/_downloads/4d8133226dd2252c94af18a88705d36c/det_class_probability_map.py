# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Tuple

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset10 as opset

from openvino_xai.common.utils import IdentityPreprocessFN
from openvino_xai.methods.white_box.base import WhiteBoxMethod


class DetClassProbabilityMap(WhiteBoxMethod):
    """
    Implements DetClassProbabilityMap, used for single-stage detectors, e.g. SSD, YOLOX or ATSS.

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
    :parameter num_anchors: Number of anchors per scale.
    :type num_anchors: List[int]
    :parameter saliency_map_size: Size of the output saliency map.
    :type saliency_map_size: Tuple[int, int] | List[int]
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        target_layer: List[str] = None,
        embed_scaling: bool = True,
        device_name: str = "CPU",
        num_anchors: List[int] | None = None,
        saliency_map_size: Tuple[int, int] | List[int] = (23, 23),
        prepare_model: bool = True,
    ):
        if target_layer is None:
            raise ValueError("target_layer is required for the detection.")
        super().__init__(model=model, preprocess_fn=preprocess_fn, embed_scaling=embed_scaling, device_name=device_name)
        self.per_class = True
        self._target_layer = target_layer
        self._num_anchors = (
            num_anchors  # Either num_anchors or num_classes has to be provided to process cls head output
        )
        self._saliency_map_size = saliency_map_size  # Not always can be obtained from model -> defined externally

        if prepare_model:
            self.prepare_model()

    def generate_xai_branch(self) -> ov.Node:
        """Implements DetClassProbabilityMap XAI algorithm."""
        cls_head_output_nodes = []
        for op in self._model_ori.get_ordered_ops():
            if op.get_friendly_name() in self._target_layer:
                cls_head_output_nodes.append(op)
        if len(cls_head_output_nodes) != len(self._target_layer):
            raise ValueError(
                f"Not all target layers found. "
                f"Expected to find {len(self._target_layer)}, found {len(cls_head_output_nodes)}."
            )

        # TODO: better handle num_classes num_anchors availability
        _, num_channels, _, _ = cls_head_output_nodes[-1].get_output_partial_shape(0)

        if self._num_anchors:
            num_cls_out_channels = num_channels.get_length() // self._num_anchors[-1]
        else:
            num_cls_out_channels = num_channels.get_length()

        if self._num_anchors:
            # Handle anchors
            for scale_idx in range(len(cls_head_output_nodes)):
                cls_scores_per_scale = cls_head_output_nodes[scale_idx]
                _, _, h, w = cls_scores_per_scale.get_output_partial_shape(0)
                cls_scores_anchor_grouped = opset.reshape(
                    cls_scores_per_scale,
                    (1, self._num_anchors[scale_idx], num_cls_out_channels, h.get_length(), w.get_length()),
                    False,
                )
                cls_scores_out = opset.reduce_max(cls_scores_anchor_grouped, 1)
                cls_head_output_nodes[scale_idx] = cls_scores_out

        # Handle scales
        for scale_idx in range(len(cls_head_output_nodes)):
            cls_head_output_nodes[scale_idx] = opset.interpolate(
                cls_head_output_nodes[scale_idx].output(0),
                output_shape=np.array([1, num_cls_out_channels, *self._saliency_map_size]),
                scales=np.array([1, 1, 1, 1], dtype=np.float32),
                mode="linear",
                shape_calculation_mode="sizes",
            )

        saliency_maps = opset.reduce_mean(opset.concat(cls_head_output_nodes, 0), 0, keep_dims=True)
        saliency_maps = opset.softmax(saliency_maps.output(0), 1)

        if self.embed_scaling:
            saliency_maps = self._scale_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps
