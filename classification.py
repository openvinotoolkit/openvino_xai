from abc import ABC
from abc import abstractmethod

import numpy as np

import openvino.runtime as ov
from openvino.runtime import opset10 as opset
from openvino.runtime import Model, Type
from openvino.preprocess import PrePostProcessor


class ExplainerBase(ABC):
    """Explainer class. Parse IR model and add XAI output node(s)."""

    def __init__(self, model_ori: ov.Model):
        self._model_ori = model_ori

    @abstractmethod
    def generate_saliency_map_node(self):
        """Implements specific XAI algorithm"""

    @staticmethod
    def get_logit_node(model):
        logit_node = (
            model.get_output_op(0)
            .input(0)
            .get_source_output()
            .get_node()
        )
        return logit_node

    def _get_last_backbone_node(self, model):
        # last_backbone_node_name = "/backbone/conv/conv.2/Div"  # mnet_v3
        # last_backbone_node_name = "/backbone/features/final_block/activate/Mul"  # effnet
        # for op in model.get_ordered_ops():
        #     if op.get_friendly_name() == last_backbone_node_name:
        #         return op

        first_head_node = self._get_first_head_node(model)
        last_backbone_node = first_head_node.input(0).get_source_output().get_node()
        return last_backbone_node

    @staticmethod
    def _get_first_head_node(model):
        # first_head_node_name = "/neck/gap/GlobalAveragePool"  # effnet and mnet_v3
        # for op in model.get_ordered_ops():
        #     if op.get_friendly_name() == first_head_node_name:
        #         return op

        for op in model.get_ordered_ops()[::-1]:
            if "GlobalAveragePool" in op.get_friendly_name():
                return op


class ActivationMapExplainer(ExplainerBase):
    """Implements ActivationMap"""

    def __init__(self, model_ori):
        super().__init__(model_ori)
        self.per_class = False

    def generate_saliency_map_node(self):
        last_backbone_node_ori = self._get_last_backbone_node(self._model_ori)
        saliency_maps = opset.reduce_mean(last_backbone_node_ori.output(0), 1)
        return saliency_maps


class ReciproCAMExplainer(ExplainerBase):
    """Implements Recipro-CAM"""

    def __init__(self, model_ori):
        super().__init__(model_ori)
        self.per_class = True

    def generate_saliency_map_node(self):
        model_clone = self._model_ori.clone()
        model_clone.get_parameters()[0].set_friendly_name('data_clone')  # for debug

        last_backbone_node_ori = self._get_last_backbone_node(self._model_ori)
        first_head_node_clone = self._get_first_head_node(model_clone)

        logit_node = self.get_logit_node(self._model_ori)
        logit_node_clone_model = self.get_logit_node(model_clone)

        logit_node.set_friendly_name("logits_ori")  # for debug
        logit_node_clone_model.set_friendly_name("logits_clone")  # for debug

        _, c, h, w = last_backbone_node_ori.get_output_partial_shape(0)
        c, h, w = c.get_length(), h.get_length(), w.get_length()

        feature_map_repeated = opset.tile(last_backbone_node_ori.output(0), (h * w, 1, 1, 1))
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


class InsertXAICls:
    """Generates a model with XAI."""

    def __init__(self, model_path: str, explain_algorithm: str = "reciprocam"):
        self._model_path = model_path
        self._model_ori = ov.Core().read_model(model_path)
        self._model_with_xai = None
        self._explain_algorithm = explain_algorithm

        self._model_ori.get_parameters()[0].set_friendly_name('data_ori')  # for debug

    @property
    def model_with_xai(self):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        return self._model_with_xai

    @property
    def model_ori(self):
        return self._model_ori

    def generate_model_with_xai(self, normalize=True):
        explainer = self._generate_explainer()
        saliency_map_node = explainer.generate_saliency_map_node()
        if normalize:
            saliency_map_node = self._normalize_saliency_maps(saliency_map_node, explainer.per_class)

        logit_node = explainer.get_logit_node(self._model_ori)

        # Just to make OTX infer/explain not to fail
        dummy_feature_vector_node = opset.constant(0, dtype=np.float32)

        model_with_xai = Model([logit_node.output(0), dummy_feature_vector_node.output(0), saliency_map_node.output(0)],
                               self._model_ori.get_parameters())
        model_with_xai = self._set_output_names_and_precisions(model_with_xai)
        self._model_with_xai = model_with_xai
        return model_with_xai

    def serialize_model_with_xai(self, model_with_xai_path):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        ov.serialize(self._model_with_xai, xml_path=model_with_xai_path)
        # TODO: Copy/rename bin. file

    def _generate_explainer(self):
        if self._explain_algorithm.lower() == "reciprocam":
            return ReciproCAMExplainer(self._model_ori)
        if self._explain_algorithm.lower() == "activationmap":
            return ActivationMapExplainer(self._model_ori)
        raise ValueError("Requested explain algorithm is not implemented.")

    @staticmethod
    def _normalize_saliency_maps(saliency_maps, per_class):
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

    @staticmethod
    def _set_output_names_and_precisions(model):
        # manually set output tensor names
        model.outputs[0].tensor.set_names({"logits"})
        model.outputs[1].tensor.set_names({"feature_vector"})
        model.outputs[2].tensor.set_names({"saliency_map"})

        # set output precisions
        ppp = PrePostProcessor(model)
        ppp.output("logits").tensor().set_element_type(Type.f32)
        ppp.output("saliency_map").tensor().set_element_type(Type.u8)
        ppp.output("feature_vector").tensor().set_element_type(Type.f32)
        model = ppp.build()
        return model
