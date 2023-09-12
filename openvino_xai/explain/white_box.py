# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List
import numpy as np

from openvino_xai.explain.base import Explainer
from openvino_xai.parameters import PostProcessParameters
from openvino_xai.saliency_map import ExplainResult, TargetExplainGroup


class WhiteBoxExplainer(Explainer):
    """Explainer explains models with XAI branch injected."""

    def explain(
            self,
            data: np.ndarray,
            target_explain_group: Optional[TargetExplainGroup] = None,
            explain_targets: Optional[List[int]] = None,
            post_processing_parameters: PostProcessParameters = PostProcessParameters(),
    ) -> ExplainResult:
        """Explain the input in white box mode.

        :param data: Data to explain.
        :type data: np.ndarray
        :param target_explain_group: Defines targets to explain: all classes, only predicted classes, etc.
        :type target_explain_group: TargetExplainGroup
        :param explain_targets: Provides list of custom targets, optional.
        :type explain_targets: Optional[List[int]]
        :param post_processing_parameters: Parameters that define post-processing.
        :type post_processing_parameters: PostProcessParameters
        """
        raw_result = self._model(data)

        if self._model.hierarchical:
            hierarchical_info = self._model.hierarchical_info["cls_heads_info"]
        else:
            hierarchical_info = None

        target_explain_group = self._get_target_explain_group(target_explain_group)
        raw_explain_result = ExplainResult(raw_result, target_explain_group, explain_targets, self._labels, hierarchical_info)

        processed_explain_result = self._get_processed_explain_result(
            raw_explain_result, data, post_processing_parameters
        )
        return processed_explain_result
