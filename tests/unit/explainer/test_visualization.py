# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino_xai.common.utils import get_min_max, scaling
from openvino_xai.explainer.explanation import Explanation
from openvino_xai.explainer.visualizer import Visualizer, colormap, overlay, resize

SALIENCY_MAPS = [
    (np.random.rand(1, 5, 5) * 255).astype(np.uint8),
    (np.random.rand(1, 2, 5, 5) * 255).astype(np.uint8),
]

EXPLAIN_ALL_CLASSES = [
    True,
    False,
]


def test_scaling_3d():
    # Test scaling on a multi-channel input
    input_saliency_map = (np.random.rand(3, 5, 5) - 0.5) * 1000
    assert (input_saliency_map < 0).any() and (input_saliency_map > 255).any()
    scaled_map = scaling(input_saliency_map)
    assert (scaled_map >= 0).all() and (scaled_map <= 255).all()


def test_scaling_2d():
    # Test scaling on a simple 2D input
    input_saliency_map = (np.random.rand(5, 5) - 0.5) * 1000
    assert (input_saliency_map < 0).any() and (input_saliency_map > 255).any()
    scaled_map = scaling(input_saliency_map)
    assert (scaled_map >= 0).all() and (scaled_map <= 255).all()


def test_scaling_cast_to_int8():
    # Test if output is correctly cast to uint8
    input_saliency_map = (np.random.rand(3, 5, 5) - 0.5) * 1000
    scaled_map = scaling(input_saliency_map)
    assert scaled_map.dtype == np.uint8

    input_saliency_map = (np.random.rand(3, 5, 5) - 0.5) * 1000
    scaled_map = scaling(input_saliency_map, cast_to_uint8=False)
    assert scaled_map.dtype == np.float32


def test_get_min_max():
    # Test min and max calculation
    input_saliency_map = np.array([[[10, 20, 30], [40, 50, 60]]]).reshape(1, -1)
    min_val, max_val = get_min_max(input_saliency_map)
    assert min_val == [10]
    assert max_val == [60]


def test_resize():
    # Test resizing functionality
    input_saliency_map = np.random.randint(0, 255, (1, 3, 3), dtype=np.uint8)
    resized_map = resize(input_saliency_map, (5, 5))
    assert resized_map.shape == (1, 5, 5)

    input_saliency_map = np.random.randint(0, 255, (2, 3, 3), dtype=np.uint8)
    resized_map = resize(input_saliency_map, (5, 5))
    assert resized_map.shape == (2, 5, 5)

    # Test resizing functionality with 700+ channels to check all classes scenario
    input_saliency_map = np.random.randint(0, 255, (1001, 3, 3), dtype=np.uint8)
    resized_map = resize(input_saliency_map, (5, 5))
    assert resized_map.shape == (1001, 5, 5)

    # Test resizing functionality for 2D saliency maps
    input_saliency_map = np.random.randint(0, 255, (3, 3), dtype=np.uint8)
    resized_map = resize(input_saliency_map, (5, 5))
    assert resized_map.shape == (5, 5)


def test_colormap():
    input_saliency_map = np.random.randint(0, 255, (1, 3, 3), dtype=np.uint8)
    colored_map = colormap(input_saliency_map)
    assert colored_map.shape == (1, 3, 3, 3)  # Check added color channels


def test_overlay():
    # Test overlay functionality
    input_image = np.ones((3, 3, 3), dtype=np.uint8) * 100
    saliency_map = np.ones((3, 3, 3), dtype=np.uint8) * 150
    overlayed_image = overlay(saliency_map, input_image)
    expected_output = np.ones((3, 3, 3), dtype=np.uint8) * 125
    assert (overlayed_image == expected_output).all()


class TestVisualizer:
    @pytest.mark.parametrize("saliency_maps", SALIENCY_MAPS)
    @pytest.mark.parametrize("explain_all_classes", EXPLAIN_ALL_CLASSES)
    @pytest.mark.parametrize("scaling", [True, False])
    @pytest.mark.parametrize("resize", [True, False])
    @pytest.mark.parametrize("colormap", [True, False])
    @pytest.mark.parametrize("overlay", [True, False])
    @pytest.mark.parametrize("overlay_weight", [0.5, 0.3])
    def test_Visualizer(
        self,
        saliency_maps,
        explain_all_classes,
        scaling,
        resize,
        colormap,
        overlay,
        overlay_weight,
    ):
        if explain_all_classes:
            explain_targets = -1
        else:
            explain_targets = [0]

        explanation = Explanation(saliency_maps, targets=explain_targets)

        raw_sal_map_dims = len(explanation.shape)
        original_input_image = np.ones((20, 20, 3))
        visualizer = Visualizer()
        explanation = visualizer(
            explanation=explanation,
            original_input_image=original_input_image,
            scaling=scaling,
            resize=resize,
            colormap=colormap,
            overlay=overlay,
            overlay_weight=overlay_weight,
        )

        assert explanation is not None
        expected_dims = raw_sal_map_dims
        if colormap or overlay:
            expected_dims += 1
        assert len(explanation.shape) == expected_dims

        if scaling and not colormap and not overlay:
            for map_ in explanation.saliency_map.values():
                assert map_.min() == 0, f"{map_.min()}"
                assert map_.max() in {254, 255}, f"{map_.max()}"
        if resize or overlay:
            for map_ in explanation.saliency_map.values():
                assert map_.shape[:2] == original_input_image.shape[:2]

        if saliency_maps.ndim == 3 and not overlay:
            explanation = Explanation(saliency_maps, targets=-1)
            visualizer = Visualizer()
            explanation_output_size = visualizer(
                explanation=explanation,
                output_size=(20, 20),
                scaling=scaling,
                resize=resize,
                colormap=colormap,
                overlay=overlay,
                overlay_weight=overlay_weight,
            )
            maps_data = explanation.saliency_map
            maps_size = explanation_output_size.saliency_map
            assert np.all(maps_data["per_image_map"] == maps_size["per_image_map"])
