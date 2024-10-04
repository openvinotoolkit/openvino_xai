# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import numpy as np
import pytest

from openvino_xai.common.parameters import Task
from openvino_xai.explainer.explanation import Explanation
from tests.unit.explainer.test_explanation_utils import VOC_NAMES

SALIENCY_MAPS = (np.random.rand(1, 20, 5, 5) * 255).astype(np.uint8)
SALIENCY_MAPS_DICT = {
    0: (np.random.rand(5, 5, 3) * 255).astype(np.uint8),
    2: (np.random.rand(5, 5, 3) * 255).astype(np.uint8),
}
SALIENCY_MAPS_DICT_EXCEPTION = {
    0: (np.random.rand(5, 5, 3, 2) * 255).astype(np.uint8),
    2: (np.random.rand(5, 5, 3, 2) * 255).astype(np.uint8),
}
SALIENCY_MAPS_IMAGE = (np.random.rand(1, 5, 5) * 255).astype(np.uint8)


class TestExplanation:
    def test_targets(self):
        explain_targets = [0, 2]
        explanation_indices = Explanation(
            SALIENCY_MAPS,
            targets=explain_targets,
            task=Task.CLASSIFICATION,
            label_names=VOC_NAMES,
        )

        explain_targets = ["aeroplane", "bird"]
        explanation_names = Explanation(
            SALIENCY_MAPS,
            targets=explain_targets,
            task=Task.CLASSIFICATION,
            label_names=VOC_NAMES,
        )

        sm_indices = explanation_indices.saliency_map
        sm_names = explanation_names.saliency_map
        assert len(sm_indices) == len(sm_names)
        assert set(sm_indices.keys()) == set(sm_names.keys()) == {0, 2}

    def test_shape(self):
        explanation = self._get_explanation()
        assert explanation.shape == (5, 5)

    def test_save(self, tmp_path):
        save_path = tmp_path / "saliency_maps"

        explanation = self._get_explanation()
        explanation.save(save_path, prefix="image_name_")
        assert os.path.isfile(save_path / "image_name_aeroplane.jpg")
        assert os.path.isfile(save_path / "image_name_bird.jpg")

        explanation = self._get_explanation()
        explanation.save(save_path)
        assert os.path.isfile(save_path / "aeroplane.jpg")
        assert os.path.isfile(save_path / "bird.jpg")

        explanation = self._get_explanation(label_names=None)
        explanation.save(save_path, postfix="_class_map")
        assert os.path.isfile(save_path / "0_class_map.jpg")
        assert os.path.isfile(save_path / "2_class_map.jpg")

        explanation = self._get_explanation()
        explanation.save(save_path, prefix="image_name_", postfix="_map")
        assert os.path.isfile(save_path / "image_name_aeroplane_map.jpg")
        assert os.path.isfile(save_path / "image_name_bird_map.jpg")

        explanation = self._get_explanation()
        explanation.save(save_path, postfix="_conf_", confidence_scores={0: 0.92, 2: 0.85})
        assert os.path.isfile(save_path / "aeroplane_conf_0.92.jpg")
        assert os.path.isfile(save_path / "bird_conf_0.85.jpg")

        explanation = self._get_explanation(saliency_maps=SALIENCY_MAPS_IMAGE, label_names=None)
        explanation.save(save_path, prefix="test_map_")
        assert os.path.isfile(save_path / "test_map_activation_map.jpg")

        explanation = self._get_explanation(saliency_maps=SALIENCY_MAPS_IMAGE, label_names=None)
        explanation.save(save_path, prefix="test_map_", postfix="_result")
        assert os.path.isfile(save_path / "test_map_activation_map_result.jpg")

    def _get_explanation(self, saliency_maps=SALIENCY_MAPS, label_names=VOC_NAMES):
        explain_targets = [0, 2]
        explanation = Explanation(
            saliency_maps,
            targets=explain_targets,
            task=Task.CLASSIFICATION,
            label_names=label_names,
        )
        return explanation

    def test_plot(self, mocker, caplog):
        explanation = self._get_explanation()

        # Invalid backend
        with pytest.raises(ValueError):
            explanation.plot(backend="invalid")

        # Plot all saliency maps
        explanation.plot()
        # Matplotloib backend
        explanation.plot([0, 2], backend="matplotlib")
        # Targets as label names
        explanation.plot(["aeroplane", "bird"], backend="matplotlib")
        # Plot all saliency maps
        explanation.plot(-1, backend="matplotlib")
        # Update the num columns for the matplotlib visualization grid
        explanation.plot(backend="matplotlib", num_columns=1)

        # Class index that is not in saliency maps will be omitted with message
        with caplog.at_level(logging.INFO):
            explanation.plot([0, 3], backend="matplotlib")
        assert "Provided class index 3 is not available among saliency maps." in caplog.text

        # Check threshold
        with caplog.at_level(logging.WARNING):
            explanation.plot([0, 2], backend="matplotlib", max_num_plots=1)

        # CV backend
        mocker.patch("cv2.imshow")
        mocker.patch("cv2.waitKey")
        explanation.plot([0, 2], backend="cv")

        # Plot activation map
        explanation = self._get_explanation(saliency_maps=SALIENCY_MAPS_IMAGE, label_names=None)
        explanation.plot()

        # Plot colored map
        explanation = self._get_explanation(saliency_maps=SALIENCY_MAPS_DICT, label_names=None)
        explanation.plot()

        # Plot wrong map shape
        with pytest.raises(Exception) as exc_info:
            explanation = self._get_explanation(saliency_maps=SALIENCY_MAPS_DICT_EXCEPTION, label_names=None)
            explanation.plot()
        assert str(exc_info.value) == "Saliency map expected to be 3 or 2-dimensional, but got 4."
