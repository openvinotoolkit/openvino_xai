import numpy as np

import openvino.runtime as ov
from openvino.runtime import opset10 as opset
from openvino.runtime import Model, Type
from openvino.preprocess import PrePostProcessor


class InsertXAI:
    """Insert inserts XAI branch into the model."""
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

        model_ori_outputs = self._explain_method.model_ori.outputs
        model_ori_params = self._explain_method.model_ori.get_parameters()
        model_with_xai = Model([*model_ori_outputs, saliency_map_node.output(0)], model_ori_params)

        saliency_map_output_id = len(model_ori_outputs)
        self._model_with_xai = self._set_output_names_and_precisions(model_with_xai, saliency_map_output_id)
        return self._model_with_xai

    @staticmethod
    def _set_output_names_and_precisions(model, saliency_map_output_id):
        model.outputs[saliency_map_output_id].tensor.set_names({"saliency_map"})
        ppp = PrePostProcessor(model)
        ppp.output("saliency_map").tensor().set_element_type(Type.u8)
        model = ppp.build()
        return model

    def serialize_model_with_xai(self, model_with_xai_path):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        ov.serialize(self._model_with_xai, xml_path=model_with_xai_path)

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
