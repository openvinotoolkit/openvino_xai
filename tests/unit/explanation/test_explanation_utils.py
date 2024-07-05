# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino_xai.explainer.utils import (
    ActivationType,
    get_explain_target_indices,
    get_score,
    is_bhwc_layout,
)

VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

LABELS_INT = [1, 4]
LABELS_STR = ["bicycle", "bottle"]


def test_get_explain_target_indices_int():
    explain_target_indices = get_explain_target_indices(LABELS_INT, VOC_NAMES)
    assert np.all(explain_target_indices == LABELS_INT)


def test_get_explain_target_indices_int_wo_names():
    explain_target_indices = get_explain_target_indices(LABELS_INT)
    assert np.all(explain_target_indices == LABELS_INT)


def test_get_explain_target_indices_str():
    explain_target_indices = get_explain_target_indices(LABELS_STR, VOC_NAMES)
    assert explain_target_indices == [1, 4]


def test_get_explain_target_indices_str_spelling():
    LABELS_STR[0] = "bicycle_"
    with pytest.raises(Exception) as exc_info:
        _ = get_explain_target_indices(LABELS_STR, VOC_NAMES)
    assert str(exc_info.value) == "No all label names found in label_names. Check spelling."


def test_get_score():
    x = np.random.rand(5)

    score = get_score(x, 0)
    assert score == x[0]

    score = get_score(x, 0, activation=ActivationType.SOFTMAX)
    x_ = np.exp(x - np.max(x))
    x_ = x_ / x_.sum()
    assert score == x_[0]

    score = get_score(x, 0, activation=ActivationType.SIGMOID)
    x_ = 1 / (1 + np.exp(-x))
    assert score == x_[0]

    x = np.random.rand(1, 5)
    score = get_score(x, 0)
    assert score == x[0][0]


def test_is_bhwc_layout():
    assert is_bhwc_layout(np.empty((1, 224, 224, 3)))
    assert is_bhwc_layout(np.empty((1, 3, 224, 224))) == False
