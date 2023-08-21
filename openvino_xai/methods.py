from abc import ABC
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

import openvino
from openvino.runtime import opset10 as opset

from openvino_xai.saliency_map import TargetExplainGroup
from openvino_xai.parse import IRParserCls


class XAIMethodBase(ABC):
    """Base class for methods that generates XAI branch of the model."""

    def __init__(self, model: openvino.runtime.Model, embed_normalization: bool = True):
        self._model_ori = model
        self._embed_normalization = embed_normalization

    @property
    def model_ori(self):
        return self._model_ori

    @property
    def model_ori_params(self):
        return self._model_ori.get_parameters()

    @abstractmethod
    def generate_xai_branch(self):
        """Implements specific XAI algorithm"""

    @staticmethod
    def _normalize_saliency_maps(saliency_maps: openvino.runtime.Node, per_class: bool) -> openvino.runtime.Node:
        """Normalize saliency maps to [0, 255] range, per-map."""
        # TODO: unify for per-class and for per-image
        if per_class:
            # Normalization for per-class saliency maps
            _, num_classes, h, w = saliency_maps.get_output_partial_shape(0)
            num_classes, h, w = num_classes.get_length(), h.get_length(), w.get_length()
            saliency_maps = opset.reshape(saliency_maps, (num_classes, h * w), False)
            max_val = opset.unsqueeze(opset.reduce_max(saliency_maps.output(0), [1]), 1)
            min_val = opset.unsqueeze(opset.reduce_min(saliency_maps.output(0), [1]), 1)
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(opset.subtract(max_val.output(0), min_val.output(0)),
                                    opset.constant(1e-12, dtype=np.float32))
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            saliency_maps = opset.reshape(saliency_maps, (1, num_classes, h, w), False)
            return saliency_maps
        else:
            # Normalization for per-image saliency map
            max_val = opset.reduce_max(saliency_maps.output(0), [0, 1, 2])
            min_val = opset.reduce_min(saliency_maps.output(0), [0, 1, 2])
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(opset.subtract(max_val.output(0), min_val.output(0)),
                                    opset.constant(1e-12, dtype=np.float32))
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            return saliency_maps


class ActivationMapXAIMethod(XAIMethodBase):
    """Implements ActivationMap"""

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: Optional[str] = None,
            embed_normalization: bool = True,
    ):
        super().__init__(model, embed_normalization)
        self.per_class = False
        self.supported_target_explain_groups = [TargetExplainGroup.IMAGE]
        self.default_target_explain_group = TargetExplainGroup.IMAGE
        self._target_layer = target_layer

    def generate_xai_branch(self) -> openvino.runtime.Node:
        output_backbone_node_ori = IRParserCls.get_output_backbone_node(self._model_ori, self._target_layer)
        saliency_maps = opset.reduce_mean(output_backbone_node_ori.output(0), 1)
        if self._embed_normalization:
            saliency_maps = self._normalize_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps


class ReciproCAMXAIMethod(XAIMethodBase):
    """Implements Recipro-CAM"""

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: Optional[str] = None,
            embed_normalization: bool = True,
    ):
        super().__init__(model, embed_normalization)
        self.per_class = True
        self.supported_target_explain_groups = [
            TargetExplainGroup.ALL_CLASSES,
            TargetExplainGroup.PREDICTED_CLASSES,
            TargetExplainGroup.CUSTOM_CLASSES,
        ]
        self.default_target_explain_group = TargetExplainGroup.ALL_CLASSES
        self._target_layer = target_layer

    def generate_xai_branch(self) -> openvino.runtime.Node:
        _model_clone = self._model_ori.clone()
        _model_clone.get_parameters()[0].set_friendly_name('data_clone')  # for debug

        # TODO: support models with multiple inputs.
        assert len(_model_clone.inputs) == 1, "Support only for models with single input."
        if not _model_clone.input(0).partial_shape[0].is_dynamic:
            partial_shape = _model_clone.input(0).partial_shape
            partial_shape[0] = -1  # make batch dimensions to be dynamic
            _model_clone.reshape(partial_shape)

        output_backbone_node_ori = IRParserCls.get_output_backbone_node(self._model_ori, self._target_layer)
        first_head_node_clone = IRParserCls.get_input_head_node(_model_clone, self._target_layer)

        logit_node = IRParserCls.get_logit_node(self._model_ori)
        logit_node_clone_model = IRParserCls.get_logit_node(_model_clone)

        # logit_node.set_friendly_name("logits_ori")  # for debug
        # logit_node_clone_model.set_friendly_name("logits_clone")  # for debug

        _, c, h, w = output_backbone_node_ori.get_output_partial_shape(0)
        c, h, w = c.get_length(), h.get_length(), w.get_length()

        feature_map_repeated = opset.tile(output_backbone_node_ori.output(0), (h * w, 1, 1, 1))
        mosaic_feature_map_mask = np.zeros((h * w, c, h, w), dtype=np.float32)
        tmp = np.arange(h * w)
        spacial_order = np.reshape(tmp, (h, w))
        for i in range(h):
            for j in range(w):
                k = spacial_order[i, j]
                mosaic_feature_map_mask[k, :, i, j] = np.ones((c))
        mosaic_feature_map_mask = opset.constant(mosaic_feature_map_mask)
        mosaic_feature_map = opset.multiply(feature_map_repeated, mosaic_feature_map_mask)

        first_head_node_clone.input(0).replace_source_output(mosaic_feature_map.output(0))

        mosaic_prediction = logit_node_clone_model

        tmp = opset.transpose(mosaic_prediction.output(0), (1, 0))
        _, num_classes = logit_node.get_output_partial_shape(0)
        saliency_maps = opset.reshape(tmp, (1, num_classes.get_length(), h, w), False)

        if self._embed_normalization:
            saliency_maps = self._normalize_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps


class DetClassProbabilityMapXAIMethod(XAIMethodBase):
    """Implements DetClassProbabilityMap, used for single-stage detectors, e.g. SSD, YOLOX or ATSS."""

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: str,
            num_anchors: int,
            saliency_map_size: Tuple[int] = (13, 13),
            embed_normalization: bool = True,
    ):
        super().__init__(model, embed_normalization)
        self.per_class = True
        self.supported_target_explain_groups = [
            TargetExplainGroup.ALL_CLASSES,
            TargetExplainGroup.PREDICTED_CLASSES,
            TargetExplainGroup.CUSTOM_CLASSES,
        ]
        self.default_target_explain_group = TargetExplainGroup.ALL_CLASSES
        self._target_layer = target_layer
        self._num_anchors = num_anchors  # Either num_anchors or num_classes has to be provided to process cls head output
        self._saliency_map_size = saliency_map_size  # Not always can be obtained from model -> defined externally

    def generate_xai_branch(self) -> openvino.runtime.Node:
        cls_head_output_nodes = []
        for op in self._model_ori.get_ordered_ops():
            if op.get_friendly_name() in self._target_layer:
                cls_head_output_nodes.append(op)
        if len(cls_head_output_nodes) != len(self._target_layer):
            raise ValueError("Not all target layers were found.")

        cls_head_output_nodes = [opset.softmax(node.output(0), 1) for node in cls_head_output_nodes]

        # TODO: better handle num_classes num_anchors availability
        _, num_channels, _, _ = cls_head_output_nodes[-1].get_output_partial_shape(0)
        num_cls_out_channels = num_channels.get_length() // self._num_anchors[-1]

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
                shape_calculation_mode="sizes"
            )
        saliency_maps = opset.reduce_mean(opset.concat(cls_head_output_nodes, 0), 0, keep_dims=True)

        if self._embed_normalization:
            saliency_maps = self._normalize_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps
