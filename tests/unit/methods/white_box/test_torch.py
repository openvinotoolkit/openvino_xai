# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copy & edit from https://github.com/openvinotoolkit/training_extensions/blob/2.1.0/tests/unit/algo/explain/test_xai_algorithms.py

import torch

from openvino_xai.methods.white_box.torch import (
    ActivationMap,
    DetClassProbabilityMap,
    ReciproCAM,
    ViTReciproCAM,
)


def test_activationmap() -> None:
    explain_algo = ActivationMap()

    assert explain_algo._norm_saliency_maps

    feature_map = torch.zeros((1, 10, 5, 5))

    saliency_maps = explain_algo.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 5, 5])


def test_reciprocam() -> None:
    def cls_head_forward_fn(_) -> None:
        return torch.zeros((25, 2))

    num_classes = 2
    optimize_gap = False
    explain_algo = ReciproCAM(
        cls_head_forward_fn,
        num_classes=num_classes,
        optimize_gap=optimize_gap,
    )

    assert explain_algo._norm_saliency_maps

    feature_map = torch.zeros((1, 10, 5, 5))

    saliency_maps = explain_algo.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 2, 5, 5])


def test_vitreciprocam() -> None:
    def cls_head_forward_fn(_) -> None:
        return torch.zeros((196, 2))

    num_classes = 2
    explain_algo = ViTReciproCAM(
        cls_head_forward_fn,
        num_classes=num_classes,
    )

    assert explain_algo._norm_saliency_maps

    feature_map = torch.zeros((1, 197, 192))

    saliency_maps = explain_algo.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 2, 14, 14])


def test_detclassprob() -> None:
    num_classes = 2
    num_anchors = [1] * 10
    explain_algo = DetClassProbabilityMap(
        num_classes=num_classes,
        num_anchors=num_anchors,
    )

    assert explain_algo._norm_saliency_maps

    backbone_out = torch.zeros((1, 5, 2, 2, 2))

    saliency_maps = explain_algo.func(backbone_out)
    assert saliency_maps.size() == torch.Size([5, 2, 2, 2])
