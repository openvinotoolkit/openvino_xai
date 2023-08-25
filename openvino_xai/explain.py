from abc import ABC
from abc import abstractmethod
from typing import Optional, List

import numpy as np
import openvino

from openvino_xai.model import XAIModel, XAIClassificationModel
from openvino_xai.parameters import ExplainParameters, PostProcessParameters
from openvino_xai.saliency_map import ExplainResult, PostProcessor, TargetExplainGroup
from openvino_xai.utils import logger


class Explainer(ABC):
    """A base interface for explainer."""

    def __init__(self, model: openvino.model_api.models.Model):
        self._model = model
        self._explain_method = self._model.explain_method if hasattr(self._model, "explain_method") else None
        self._labels = self._model.labels

    @abstractmethod
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError

    def _get_target_explain_group(self, target_explain_group):
        if target_explain_group:
            if self._explain_method:
                assert target_explain_group in self._explain_method.supported_target_explain_groups, \
                    f"Provided target_explain_group {target_explain_group} is not supported by the explain method."
            return target_explain_group
        else:
            if self._explain_method:
                return self._explain_method.default_target_explain_group
            else:
                raise ValueError("Model with XAI branch was created outside of Openvino-XAI library. "
                                 "Please explicitly provide target_explain_group to the explain call.")

    @staticmethod
    def _get_processed_explain_result(raw_explain_result, data, post_processing_parameters):
        if post_processing_parameters:
            post_processor = PostProcessor(
                raw_explain_result,
                data,
                post_processing_parameters.normalize,
                post_processing_parameters.resize,
                post_processing_parameters.colormap,
                post_processing_parameters.overlay,
                post_processing_parameters.overlay_weight,
            )
        else:
            post_processor = PostProcessor(raw_explain_result, data)
        processed_explain_result = post_processor.postprocess()
        return processed_explain_result

class WhiteBoxExplainer(Explainer):
    """Explainer explains models with XAI branch injected."""

    def explain(
            self,
            data: np.ndarray,
            target_explain_group: Optional[TargetExplainGroup] = None,
            explain_targets: Optional[List[int]] = None,
            post_processing_parameters: Optional[PostProcessParameters] = None,
    ) -> ExplainResult:
        """Explain the input in white box mode."""
        raw_result = self._model(data)

        target_explain_group = self._get_target_explain_group(target_explain_group)
        raw_explain_result = ExplainResult(raw_result, target_explain_group, explain_targets, self._labels)

        processed_explain_result = self._get_processed_explain_result(
            raw_explain_result, data, post_processing_parameters
        )
        return processed_explain_result


class BlackBoxExplainer(Explainer):
    """Base class for explainers that consider model as a black-box."""


class RISEExplainer(BlackBoxExplainer):
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError


class DRISEExplainer(BlackBoxExplainer):
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError


class AutoExplainer(Explainer):
    """Explain in auto mode, using white box or black box approach."""

    def __init__(self, model: openvino.model_api.models.Model, explain_parameters: Optional[ExplainParameters] = None):
        super().__init__(model)
        self._explain_parameters = explain_parameters


class ClassificationAutoExplainer(AutoExplainer):
    """Explain classification models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray, target_explain_group: Optional[TargetExplainGroup] = None) -> ExplainResult:
        """
        Implements three explain scenarios, for different IR models:
            1. IR model contain xai branch -> infer Model API wrapper.
            2. If not (1), IR model can be augmented with XAI branch -> augment and infer.
            3. If not (1) and (2), IR model can NOT be augmented with XAI branch -> use XAI BB method.

        Args:
            data(numpy.ndarray): data to explain.
            target_explain_group(TargetExplainGroup): Target explain group.
        """
        if XAIModel.has_xai(self._model.inference_adapter.model):
            logger.info("Model already has XAI - using White Box explainer.")
            explanations = WhiteBoxExplainer(self._model).explain(data, target_explain_group)
            return explanations
        else:
            try:
                logger.info("Model does not have XAI - trying to insert XAI and use White Box explainer.")
                self._model = XAIClassificationModel.insert_xai(self._model, self._explain_parameters)
                explanations = WhiteBoxExplainer(self._model).explain(data)
                return explanations
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model. Calling Black Box explainer.")
                explanations = RISEExplainer(self._model).explain(data)
                return explanations


class DetectionAutoExplainer(AutoExplainer):
    """Explain detection models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray) -> np.ndarray:
        """Explain the input."""
        raise NotImplementedError
