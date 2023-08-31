# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Optional, List
import numpy as np

from openvino_xai.saliency_map import ExplainResult, PostProcessor, TargetExplainGroup
from . import Explainer


class WhiteBoxExplainer(Explainer):
    """Explainer explains models with XAI branch injected."""

    def explain(
        self,
        data: np.ndarray,
        target_explain_group: Optional[TargetExplainGroup] = None,
        explain_targets: Optional[List[int]] = None,
        post_processing_parameters: Optional[Dict[str, Any]] = None,
    ) -> ExplainResult:
        """Explain the input in white box mode."""
        raw_result = self._model(data)

        target_explain_group = self._get_target_explain_group(target_explain_group)
        explain_result = ExplainResult(raw_result, target_explain_group, explain_targets, self._labels)

        post_processing_parameters = post_processing_parameters or {}
        post_processor = PostProcessor(explain_result, data, **post_processing_parameters)
        explain_result = post_processor.postprocess()
        return explain_result