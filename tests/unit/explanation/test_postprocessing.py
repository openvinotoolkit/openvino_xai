# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino_xai.explanation import normalize, get_min_max, resize, colormap, overlay


def test_normalize():
    # Test normalization on a simple multi-channel input
    input_saliency_map = (np.random.rand(3, 5, 5) - 0.5) * 1000
    assert (input_saliency_map < 0).any() and (input_saliency_map > 255).any()
    normalized_map = normalize(input_saliency_map)
    assert (normalized_map >= 0).all() and (normalized_map <= 255).all()


def test_normalize_cast_to_int8():
    # Test if output is correctly cast to uint8
    input_saliency_map = (np.random.rand(3, 5, 5) - 0.5) * 1000
    normalized_map = normalize(input_saliency_map)
    assert normalized_map.dtype == np.uint8


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
