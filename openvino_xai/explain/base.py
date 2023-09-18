# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod

import numpy as np
import openvino

from openvino_xai.saliency_map import ExplainResult, PostProcessor


class Explainer(ABC):
    """A base interface for explainer."""

    def __init__(self, model: openvino.model_api.models.Model):
        self._model = model
        self._explain_method = self._model.explain_method if hasattr(self._model, "explain_method") else None
        self._labels = self._model.labels

    @abstractmethod
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        # TODO: handle path_to_data as input as well?
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
                raise ValueError("Please explicitly provide target_explain_group to the explain call.")

    @staticmethod
    def _get_processed_explain_result(raw_explain_result, data, post_processing_parameters):
        post_processor = PostProcessor(
            raw_explain_result,
            data,
            post_processing_parameters,
        )
        processed_explain_result = post_processor.postprocess()
        return processed_explain_result
