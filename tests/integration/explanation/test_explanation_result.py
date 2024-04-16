# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino_xai.explanation import TargetExplainGroup
from openvino_xai.explanation.explanation_result import ExplanationResult
from tests.unit.explanation.test_utils import VOC_NAMES

SALIENCY_MAPS = (np.random.rand(1, 20, 5, 5) * 255).astype(np.uint8)


class TestExplanationResult:
    def test_target_explain_labels(self):
        explain_targets = [0, 2]
        explanation_result_indices = ExplanationResult(
            SALIENCY_MAPS,
            target_explain_group=TargetExplainGroup.CUSTOM,
            target_explain_labels=explain_targets,
            label_names=VOC_NAMES,
        )

        explain_targets = ["aeroplane", "bird"]
        explanation_result_names = ExplanationResult(
            SALIENCY_MAPS,
            target_explain_group=TargetExplainGroup.CUSTOM,
            target_explain_labels=explain_targets,
            label_names=VOC_NAMES,
        )

        sm_indices = explanation_result_indices.saliency_map
        sm_names = explanation_result_names.saliency_map
        assert len(sm_indices) == len(sm_names)
        assert set(sm_indices.keys()) == set(sm_names.keys()) == {0, 2}
