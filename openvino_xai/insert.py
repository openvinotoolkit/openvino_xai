from abc import ABC
from abc import abstractmethod

import numpy as np

import openvino.runtime as ov
from openvino.runtime import opset10 as opset
from openvino.runtime import Model, Type
from openvino.preprocess import PrePostProcessor

from openvino_xai.parse import IRParser


class InsertXAIBase(ABC):
    def __init__(self, explain_method):
        self._model_with_xai = None
        self._explain_method = explain_method

    @property
    def model_with_xai(self):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        return self._model_with_xai

    def generate_model_with_xai(self, normalize=True):
        """Generates model with XAI inserted."""
        saliency_map_node = self._explain_method.generate_xai_branch()
        if normalize:
            saliency_map_node = self._normalize_saliency_maps(saliency_map_node, self._explain_method.per_class)

        output_nodes = self._get_main_output_nodes(self._explain_method.model_ori)

        model_with_xai = Model([*output_nodes, saliency_map_node.output(0)], self._explain_method.model_ori_params)
        self._model_with_xai = self._set_output_names_and_precisions(model_with_xai)
        return self._model_with_xai

    @staticmethod
    @abstractmethod
    def _get_main_output_nodes(model):
        pass

    @staticmethod
    @abstractmethod
    def _set_output_names_and_precisions(model):
        pass

    def serialize_model_with_xai(self, model_with_xai_path):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        ov.serialize(self._model_with_xai, xml_path=model_with_xai_path)
        # TODO: Copy/rename bin. file?

    @staticmethod
    def _normalize_saliency_maps(saliency_maps, per_class):
        # TODO: should be implemented in the model wrapper?
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


class InsertXAICls(InsertXAIBase):
    """Generates a classification model with XAI."""

    @staticmethod
    def _get_main_output_nodes(model):
        # All outputs except sal_map can be removed, keep it now for debug
        logit_node = IRParser.get_logit_node(model)
        # Just to make OTX infer/explain not to fail
        dummy_feature_vector_node = opset.constant(0, dtype=np.float32)
        return logit_node.output(0), dummy_feature_vector_node.output(0)

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


class InsertXAIDet(InsertXAIBase):
    """Generates a detection model with XAI."""

    @staticmethod
    def _get_main_output_nodes(model):
        # All outputs except sal_map can be removed, keep it now for debug
        boxes_node = IRParser.get_logit_node(model, 0)
        labels_node = IRParser.get_logit_node(model, 1)
        # Just to make OTX infer/explain not to fail
        dummy_feature_vector_node = opset.constant(1, dtype=np.float32)
        return boxes_node.output(0), labels_node.output(0), dummy_feature_vector_node.output(0)

    @staticmethod
    def _set_output_names_and_precisions(model):
        # manually set output tensor names
        model.outputs[0].tensor.set_names({"boxes"})
        model.outputs[1].tensor.set_names({"labels"})
        model.outputs[2].tensor.set_names({"feature_vector"})
        model.outputs[3].tensor.set_names({"saliency_map"})

        # set output precisions
        ppp = PrePostProcessor(model)
        ppp.output("boxes").tensor().set_element_type(Type.f32)
        ppp.output("labels").tensor().set_element_type(Type.i64)
        ppp.output("saliency_map").tensor().set_element_type(Type.u8)
        ppp.output("feature_vector").tensor().set_element_type(Type.f32)
        model = ppp.build()
        return model
