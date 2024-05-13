# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino_xai.explanation.utils import get_explain_target_indices

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
    assert explain_target_indices == LABELS_INT


def test_get_explain_target_indices_int_wo_names():
    explain_target_indices = get_explain_target_indices(LABELS_INT)
    assert explain_target_indices == LABELS_INT


def test_get_explain_target_indices_str():
    explain_target_indices = get_explain_target_indices(LABELS_STR, VOC_NAMES)
    assert explain_target_indices == [1, 4]


def test_get_explain_target_indices_str_spelling():
    LABELS_STR[0] = "bicycle_"
    with pytest.raises(Exception) as exc_info:
        _ = get_explain_target_indices(LABELS_STR, VOC_NAMES)
    assert str(exc_info.value) == "No all label names found in label_names. Check spelling."
