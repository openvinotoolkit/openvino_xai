# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from openvino_xai.explanation import TargetExplainGroup
from openvino_xai.explanation.explanation_result import ExplanationResult
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

SALIENCY_MAPS = (np.random.rand(1, 20, 5, 5) * 255).astype(np.uint8)
SALIENCY_MAPS_IMAGE = (np.random.rand(1, 5, 5) * 255).astype(np.uint8)


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

    def test_target_explain_group_image(self):
        explanation_result = ExplanationResult(
            SALIENCY_MAPS_IMAGE,
            target_explain_group=TargetExplainGroup.IMAGE,
        )
        assert "per_image_map" in explanation_result.saliency_map
        assert len(explanation_result.saliency_map) == 1
        assert explanation_result.saliency_map["per_image_map"].shape == (5, 5)

    def test_sal_map_shape(self):
        explanation_result = self._get_explanation_result()
        assert explanation_result.sal_map_shape == (5, 5)

    def test_save(self, tmp_path):
        save_path = tmp_path / "saliency_maps"

        explanation_result = self._get_explanation_result()
        explanation_result.save(save_path, "test_map")
        assert os.path.isfile(save_path / "test_map_target_aeroplane.jpg")
        assert os.path.isfile(save_path / "test_map_target_bird.jpg")

        explanation_result = self._get_explanation_result(label_names=None)
        explanation_result.save(save_path, "test_map")
        assert os.path.isfile(save_path / "test_map_target_0.jpg")
        assert os.path.isfile(save_path / "test_map_target_2.jpg")

        explanation_result = self._get_explanation_result(saliency_maps=SALIENCY_MAPS_IMAGE, label_names=None)
        explanation_result.save(save_path, "test_map")
        assert os.path.isfile(save_path / "test_map.jpg")

    def _get_explanation_result(self, saliency_maps=SALIENCY_MAPS, label_names=VOC_NAMES):
        explain_targets = [0, 2]
        explanation_result = ExplanationResult(
            saliency_maps,
            target_explain_group=TargetExplainGroup.CUSTOM,
            target_explain_labels=explain_targets,
            label_names=label_names,
        )
        return explanation_result
