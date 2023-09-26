# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod
from typing import Optional
from pathlib import Path
import os

import openvino
from openvino.model_api.models import ClassificationModel
from openvino.model_api.models import DetectionModel

from openvino_xai.insert import InsertXAI
from openvino_xai.methods import (
    ReciproCAMXAIMethod,
    ActivationMapXAIMethod,
    DetClassProbabilityMapXAIMethod,
    XAIMethodBase,
)
from openvino_xai.parameters import (
    ExplainParameters,
    ClassificationExplainParametersWB,
    DetectionExplainParametersWB,
    XAIMethodType,
)
from openvino_xai.utils import logger


class XAIModel(ABC):
    """Factory for creating Model API model wrapper and updating it with XAI functionality."""

    @classmethod
    def create_model(cls, *args, **kwargs) -> openvino.model_api.models.Model:
        """Creates Model API model wrapper with XAI branch."""
        explain_parameters = kwargs.pop("explain_parameters", None)

        model_api_model_class = cls._get_model_api_model_class()
        model_api_wrapper = model_api_model_class.create_model(*args, **kwargs)
        logger.info("Created Model API wrapper.")

        if cls.has_xai(model_api_wrapper.inference_adapter.model):
            logger.info("Provided model already contains XAI branch, return it as-is.")
            return model_api_wrapper

        path_to_model = args[0]
        model_suffix = Path(path_to_model).suffix
        assert model_suffix == ".xml", f"XAI can be inserted only into OV IR, " \
                                       f"but provided model has {model_suffix} extension. " \
                                       f"Please provide path to OV IR for white-box explain methods."
        model_api_wrapper = cls.insert_xai(model_api_wrapper, explain_parameters)
        return model_api_wrapper

    @classmethod
    def insert_xai(
            cls,
            model_api_wrapper: openvino.model_api.models.Model,
            explain_parameters: ExplainParameters,
    ) -> openvino.model_api.models.Model:
        """
        Insert XAI into IR model stored in Model API wrapper.

        :param model_api_wrapper: Original ModelAPI wrapper.
        :type model_api_wrapper: openvino.model_api.models.Model
        :param explain_parameters: Explain parameters that parametrize explain method,
            that will be inserted into the model graph.
        :type explain_parameters: ExplainParameters
        :return: Modified ModelAPI wrapper with XAI head.
        """
        # Insert XAI branch into the model
        model_ir = model_api_wrapper.get_model()
        explain_method = cls.generate_explain_method(model_ir, explain_parameters)
        xai_generator = InsertXAI(explain_method)
        model_ir_with_xai = xai_generator.generate_model_with_xai()

        # Update Model API wrapper
        model_api_wrapper.explain_method = explain_method
        model_api_wrapper.inference_adapter.model = model_ir_with_xai
        if hasattr(model_api_wrapper, "out_layer_names"):
            model_api_wrapper.out_layer_names.append("saliency_map")
        if model_api_wrapper.model_loaded:
            model_api_wrapper.load(force=True)

        assert cls.has_xai(model_api_wrapper.inference_adapter.model), (
            "Insertion of the XAI branch into the model " "was not successful."
        )
        logger.info("Insertion of the XAI branch into the model was successful.")
        return model_api_wrapper

    @classmethod
    def insert_xai_into_native_ir(
            cls,
            model_path: str,
            output: Optional[str] = None,
            explain_parameters: Optional[ExplainParameters] = None,
    ) -> openvino.runtime.Model:
        """Insert XAI directly into IR model.

        :param model_path: Path to OV IR model.
        :type model_path: str

        :param output: Output path where to save updated OV IR model.
        :type output: Optional[str]

        :param explain_parameters: Explain parameters that parametrize explain method,
            that will be inserted into the model graph.
        :type explain_parameters: Optional[ExplainParameters]

        :return: Modified OV IR model with XAI head.
        """
        model_name = Path(model_path).stem
        model_ir = openvino.runtime.Core().read_model(model_path)
        if cls.has_xai(model_ir):
            logger.info("Provided IR model already contains XAI branch, return it as-is.")
            return model_ir

        explain_method = cls.generate_explain_method(model_ir, explain_parameters)
        xai_generator = InsertXAI(explain_method)
        model_with_xai = xai_generator.generate_model_with_xai()
        assert cls.has_xai(model_with_xai), "Insertion of the XAI branch into the model was not successful."
        logger.info("Insertion of the XAI branch into the model was successful.")
        if output:
            xai_generator.serialize_model_with_xai(os.path.join(output, model_name + "_xai.xml"))
        return model_with_xai

    @classmethod
    @abstractmethod
    def _get_model_api_model_class(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_explain_method(
            cls, model_ir: openvino.runtime.Model, explain_parameters: ExplainParameters
    ) -> XAIMethodBase:
        """Generates instance of the explain method class.

        :param model: OV IR model.
        :type model: openvino.runtime.Model
        :param explain_parameters: Explain parameters that parametrize explain method,
            that will be inserted into the model graph.
        :type explain_parameters: ExplainParameters
        """
        raise NotImplementedError

    @staticmethod
    def has_xai(model: openvino.runtime.Model) -> bool:
        """Check if the model contain XAI.

        :param model: OV IR model.
        :type model: openvino.runtime.Model
        :return: True is the model has XAI head, False otherwise.
        """
        if not isinstance(model, openvino.runtime.Model):
            raise ValueError(f"Input model has to be openvino.runtime.Model instance, but got{type(model)}.")
        for output in model.outputs:
            if "saliency_map" in output.get_names():
                return True
        return False


class XAIClassificationModel(XAIModel):
    """Creates classification Model API model wrapper."""

    @classmethod
    def _get_model_api_model_class(cls):
        return ClassificationModel

    @classmethod
    def generate_explain_method(
            cls, model: openvino.runtime.Model, explain_parameters: Optional[ClassificationExplainParametersWB] = None
    ) -> XAIMethodBase:
        """Generates instance of the classification explain method class.

        :param model: OV IR model.
        :type model: openvino.runtime.Model
        :param explain_parameters: Explain parameters that parametrize explain method,
            that will be inserted into the model graph.
        :type explain_parameters: ExplainParameters
        """
        if explain_parameters is None:
            return ReciproCAMXAIMethod(model)

        explain_method_type = explain_parameters.explain_method_type
        if explain_method_type == XAIMethodType.RECIPROCAM:
            return ReciproCAMXAIMethod(
                model, explain_parameters.target_layer, explain_parameters.embed_normalization
            )
        if explain_method_type == XAIMethodType.ACTIVATIONMAP:
            return ActivationMapXAIMethod(
                model, explain_parameters.target_layer, explain_parameters.embed_normalization
            )
        raise ValueError(f"Requested explain method {explain_method_type} is not implemented.")


class XAIDetectionModel(XAIModel):
    """Creates detection Model API model wrapper."""

    @classmethod
    def _get_model_api_model_class(cls):
        return DetectionModel

    @classmethod
    def generate_explain_method(
            cls, model: openvino.runtime.Model, explain_parameters: DetectionExplainParametersWB
    ) -> XAIMethodBase:
        """Generates instance of the detection explain method class.

        :param model: OV IR model.
        :type model: openvino.runtime.Model
        :param explain_parameters: Explain parameters that parametrize explain method,
            that will be inserted into the model graph.
        :type explain_parameters: ExplainParameters
        """
        if explain_parameters is None:
            raise ValueError("explain_parameters is required for the detection models.")

        explain_method_type = explain_parameters.explain_method_type
        if explain_method_type == XAIMethodType.DETCLASSPROBABILITYMAP:
            return DetClassProbabilityMapXAIMethod(
                model,
                explain_parameters.target_layer,
                explain_parameters.num_anchors,
                explain_parameters.saliency_map_size,
                explain_parameters.embed_normalization,
            )
        raise ValueError(f"Requested explain method {explain_method_type} is not implemented.")
