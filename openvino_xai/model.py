from abc import ABC
from abc import abstractmethod

from openvino.model_api.models import ClassificationModel
from openvino.model_api.models import DetectionModel

from openvino_xai.insert import InsertXAI
from openvino_xai.methods import ReciproCAMXAIMethod, ActivationMapXAIMethod, DetClassProbabilityMapXAIMethod
from openvino_xai.utils import logger


class XAIModel(ABC):
    """Factory for creating Model API model wrapper and updating it with XAI functionality."""

    @classmethod
    def create_model(cls, *args, **kwargs):
        """Creates Model API model wrapper with XAI branch."""
        explain_parameters = kwargs.pop("explain_parameters", None)

        model_api_model_class = cls._get_model_api_model_class()
        model_api_wrapper = model_api_model_class.create_model(*args, **kwargs)
        logger.info("Created Model API wrapper.")

        if cls.has_xai(model_api_wrapper):
            logger.info("Provided IR model already contains XAI branch, return it as-is.")
            return model_api_wrapper

        model_api_wrapper = cls.insert_xai(model_api_wrapper, explain_parameters)
        return model_api_wrapper

    @classmethod
    def insert_xai(cls, model_api_wrapper, explain_parameters):
        # Insert XAI branch into the model
        model_ir = model_api_wrapper.get_model()
        explain_method = cls._generate_explain_method(model_ir, explain_parameters)
        xai_generator = InsertXAI(explain_method)
        model_ir_with_xai = xai_generator.generate_model_with_xai()
        logger.info(f"Original model:\n{explain_method.model_ori}")
        logger.info(f"Model with XAI inserted:\n{model_ir_with_xai}")

        # Update Model API wrapper
        model_api_wrapper._explain_method = explain_method
        model_api_wrapper.inference_adapter.model = model_ir_with_xai
        if hasattr(model_api_wrapper, "out_layer_names"):
            model_api_wrapper.out_layer_names.append("saliency_map")
        if model_api_wrapper.model_loaded:
            model_api_wrapper.load(force=True)

        assert cls.has_xai(model_api_wrapper), "Insertion of the XAI branch into the model was not successful."
        logger.info("Insertion of the XAI branch into the model was successful.")
        return model_api_wrapper

    @classmethod
    @abstractmethod
    def _get_model_api_model_class(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _generate_explain_method(cls, model_ir, explain_parameters):
        raise NotImplementedError

    @staticmethod
    def has_xai(model):
        """Check if the model contain XAI."""
        for output in model.inference_adapter.model.outputs:
            if "saliency_map" in output.get_names():
                return True
        return False


class XAIClassificationModel(XAIModel):
    """Creates classification Model API model wrapper."""

    @classmethod
    def _get_model_api_model_class(cls):
        return ClassificationModel

    @classmethod
    def _generate_explain_method(cls, model, explain_parameters):
        if explain_parameters is None:
            return ReciproCAMXAIMethod(model)

        explain_method_name = explain_parameters.pop("explain_method_name", None)
        if explain_method_name is None or explain_method_name.lower() == "reciprocam":
            return ReciproCAMXAIMethod(model, **explain_parameters)
        if explain_method_name.lower() == "activationmap":
            return ActivationMapXAIMethod(model, **explain_parameters)
        raise ValueError(f"Requested explain method {explain_method_name} is not implemented.")


class XAIDetectionModel(XAIModel):
    """Creates detection Model API model wrapper."""

    @classmethod
    def _get_model_api_model_class(cls):
        return DetectionModel

    @classmethod
    def _generate_explain_method(cls, model, explain_parameters):
        if explain_parameters is None:
            raise ValueError("explain_parameters is required for the detection models.")

        explain_method_name = explain_parameters.pop("explain_method_name")
        if explain_method_name is None:
            raise ValueError("explain_method_name is required for the detection models.")
        if explain_method_name.lower() == "detclassprobabilitymap":
            return DetClassProbabilityMapXAIMethod(model, **explain_parameters)
        raise ValueError(f"Requested explain method {explain_method_name} is not implemented.")
