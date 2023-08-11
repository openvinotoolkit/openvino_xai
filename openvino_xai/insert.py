import openvino
import openvino.runtime as ov
from openvino.runtime import Model, Type
from openvino.preprocess import PrePostProcessor

from openvino_xai.methods import XAIMethodBase


class InsertXAI:
    """Insert inserts XAI branch into the model."""
    def __init__(self, explain_method: XAIMethodBase):
        self._model_with_xai = None
        self._explain_method = explain_method

    @property
    def model_with_xai(self):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        return self._model_with_xai

    def generate_model_with_xai(self) -> openvino.runtime.Model:
        """Generates model with XAI inserted."""
        saliency_map_node = self._explain_method.generate_xai_branch()

        model_ori_outputs = self._explain_method.model_ori.outputs
        model_ori_params = self._explain_method.model_ori.get_parameters()
        model_with_xai = Model([*model_ori_outputs, saliency_map_node.output(0)], model_ori_params)

        saliency_map_output_id = len(model_ori_outputs)
        self._model_with_xai = self._set_output_names_and_precisions(model_with_xai, saliency_map_output_id)
        return self._model_with_xai

    @staticmethod
    def _set_output_names_and_precisions(
            model: openvino.runtime.Model, saliency_map_output_id: int
    ) -> openvino.runtime.Model:
        model.outputs[saliency_map_output_id].tensor.set_names({"saliency_map"})
        ppp = PrePostProcessor(model)
        ppp.output("saliency_map").tensor().set_element_type(Type.u8)
        model = ppp.build()
        return model

    def serialize_model_with_xai(self, model_with_xai_path: openvino.runtime.Model):
        if not self._model_with_xai:
            raise RuntimeError("First, generate model with xai.")
        ov.serialize(self._model_with_xai, xml_path=model_with_xai_path)
