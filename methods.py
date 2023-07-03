from abc import ABC
from abc import abstractmethod

import numpy as np

import openvino.runtime as ov
from openvino.runtime import opset10 as opset


class XAIMethodBase(ABC):
    """Explainer class. Parse IR model and add XAI output node(s)."""

    def __init__(self, model_ori: ov.Model):
        self._model_ori = model_ori

    @abstractmethod
    def generate_saliency_map_node(self):
        """Implements specific XAI algorithm"""

    @staticmethod
    def get_logit_node(model, output_id=0):
        logit_node = (
            model.get_output_op(output_id)
            .input(0)
            .get_source_output()
            .get_node()
        )
        return logit_node

    def _get_output_backbone_node(self, model):
        # output_backbone_node_name = "/backbone/conv/conv.2/Div"  # mnet_v3
        # output_backbone_node_name = "/backbone/features/final_block/activate/Mul"  # effnet
        # for op in model.get_ordered_ops():
        #     if op.get_friendly_name() == output_backbone_node_name:
        #         return op

        first_head_node = self._get_first_head_node(model)
        output_backbone_node = first_head_node.input(0).get_source_output().get_node()
        return output_backbone_node

    @staticmethod
    def _get_first_head_node(model):
        # first_head_node_name = "/neck/gap/GlobalAveragePool"  # effnet and mnet_v3
        # for op in model.get_ordered_ops():
        #     if op.get_friendly_name() == first_head_node_name:
        #         return op

        for op in model.get_ordered_ops()[::-1]:
            if "GlobalAveragePool" in op.get_friendly_name():
                return op


class ActivationMapXAIMethod(XAIMethodBase):
    """Implements ActivationMap"""

    def __init__(self, model_ori):
        super().__init__(model_ori)
        self.per_class = False

    def generate_saliency_map_node(self):
        output_backbone_node_ori = self._get_output_backbone_node(self._model_ori)
        saliency_maps = opset.reduce_mean(output_backbone_node_ori.output(0), 1)
        return saliency_maps


class ReciproCAMXAIMethod(XAIMethodBase):
    """Implements Recipro-CAM"""

    def __init__(self, model_ori):
        super().__init__(model_ori)
        self.per_class = True

    def generate_saliency_map_node(self):
        model_clone = self._model_ori.clone()
        model_clone.get_parameters()[0].set_friendly_name('data_clone')  # for debug

        output_backbone_node_ori = self._get_output_backbone_node(self._model_ori)
        first_head_node_clone = self._get_first_head_node(model_clone)

        logit_node = self.get_logit_node(self._model_ori)
        logit_node_clone_model = self.get_logit_node(model_clone)

        logit_node.set_friendly_name("logits_ori")  # for debug
        logit_node_clone_model.set_friendly_name("logits_clone")  # for debug

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
        return saliency_maps


class DetClassProbabilityMapXAIMethod(XAIMethodBase):
    """Implements DetClassProbabilityMap"""

    def __init__(self, model_ori, saliency_map_size=(13, 13)):
        super().__init__(model_ori)
        self.per_class = True
        self._num_anchors = [1] * 10
        self._saliency_map_size = saliency_map_size

    def generate_saliency_map_node(self):
        # # YOLOX
        # classification_head_output_node_names = [
        #     "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
        #     "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
        #     "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
        # ]

        # ATSS
        classification_head_output_node_names = [
            "/bbox_head/atss_cls_0/Conv/WithoutBiases",
            "/bbox_head/atss_cls_1/Conv/WithoutBiases",
            "/bbox_head/atss_cls_2/Conv/WithoutBiases",
            "/bbox_head/atss_cls_3/Conv/WithoutBiases",
            "/bbox_head/atss_cls_4/Conv/WithoutBiases",
        ]

        cls_head_output_nodes = []
        for op in self._model_ori.get_ordered_ops():
            if op.get_friendly_name() in classification_head_output_node_names:
                cls_head_output_nodes.append(op)

        cls_head_output_nodes = [opset.softmax(node.output(0), 1) for node in cls_head_output_nodes]

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

        # saliency_maps = opset.softmax(cls_head_output_nodes[-1].output(0), 1)
        # saliency_maps = opset.multiply(saliency_maps, opset.constant(255, dtype=np.float32))
        return saliency_maps
