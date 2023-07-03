from abc import ABC
from abc import abstractmethod

import numpy as np

import openvino.runtime as ov
from openvino.runtime import opset10 as opset
from openvino.runtime import Model, Type
from openvino.preprocess import PrePostProcessor

from openvino_xai.methods import ReciproCAMXAIMethod, ActivationMapXAIMethod, DetClassProbabilityMapXAIMethod


class InsertXAIBase(ABC):
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model_ori = ov.Core().read_model(model_path)
        self._model_with_xai = None

        self._model_ori.get_parameters()[0].set_friendly_name('data_ori')  # for debug

    @property
    def model_with_xai(self):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        return self._model_with_xai

    @property
    def model_ori(self):
        return self._model_ori

    @abstractmethod
    def generate_model_with_xai(self):
        """Generates model with XAI inserted."""

    def serialize_model_with_xai(self, model_with_xai_path):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        ov.serialize(self._model_with_xai, xml_path=model_with_xai_path)
        # TODO: Copy/rename bin. file

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


class InsertXAICls(InsertXAIBase):
    """Generates a classification model with XAI."""

    def __init__(self, model_path: str, explain_algorithm: str = "reciprocam"):
        super().__init__(model_path)
        self._explain_algorithm = explain_algorithm
        self._explain_params = None

    def generate_model_with_xai(self, normalize=True):
        explainer = self._generate_explain_method()
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

    def _generate_explain_method(self):
        if self._explain_algorithm.lower() == "reciprocam":
            return ReciproCAMXAIMethod(self._model_ori)
        if self._explain_algorithm.lower() == "activationmap":
            return ActivationMapXAIMethod(self._model_ori)
        raise ValueError("Requested explain algorithm is not implemented.")

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

    def __init__(self, model_path: str, explain_algorithm: str = "detclassprobabilitymap"):
        super().__init__(model_path)
        self._explain_algorithm = explain_algorithm
        self._explain_params = None

    def generate_model_with_xai(self, normalize=True):
        explainer = self._generate_explain_method()
        saliency_map_node = explainer.generate_saliency_map_node()
        if normalize:
            saliency_map_node = self._normalize_saliency_maps(saliency_map_node, explainer.per_class)

        boxes_node = explainer.get_logit_node(self._model_ori, 0)
        labels_node = explainer.get_logit_node(self._model_ori, 1)

        # Just to make OTX infer/explain not to fail
        dummy_feature_vector_node = opset.constant(1, dtype=np.float32)

        model_with_xai = Model([boxes_node.output(0), labels_node.output(0), dummy_feature_vector_node.output(0),
                                saliency_map_node.output(0)], self._model_ori.get_parameters())
        model_with_xai = self._set_output_names_and_precisions(model_with_xai)
        self._model_with_xai = model_with_xai
        return model_with_xai

    def _generate_explain_method(self):
        if self._explain_algorithm.lower() == "detclassprobabilitymap":
            return DetClassProbabilityMapXAIMethod(self._model_ori)
        raise ValueError("Requested explain algorithm is not implemented.")

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
