# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino_xai.explanation.explanation_parameters import PostProcessParameters, TargetExplainGroup
from openvino_xai.explanation.explanation_result import ExplanationResult
from openvino_xai.explanation.post_process import PostProcessor


SALIENCY_MAPS = [
    (np.random.rand(1, 5, 5) * 255).astype(np.uint8),
    (np.random.rand(1, 2, 5, 5) * 255).astype(np.uint8),
]

TARGET_EXPLAIN_GROUPS = [
    TargetExplainGroup.ALL,
    TargetExplainGroup.CUSTOM,
]


class TestPostProcessor:
    # TODO: Create a unit test for each postprocess method

    @pytest.mark.parametrize("saliency_maps", SALIENCY_MAPS)
    @pytest.mark.parametrize("target_explain_group", TARGET_EXPLAIN_GROUPS)
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("resize", [True, False])
    @pytest.mark.parametrize("colormap", [True, False])
    @pytest.mark.parametrize("overlay", [True, False])
    @pytest.mark.parametrize("overlay_weight", [0.5, 0.3])
    def test_postprocessor(
        self,
        saliency_maps,
        target_explain_group,
        normalize,
        resize,
        colormap,
        overlay,
        overlay_weight,
    ):
        post_processing_parameters = PostProcessParameters(
            normalize=normalize,
            resize=resize,
            colormap=colormap,
            overlay=overlay,
            overlay_weight=overlay_weight,
        )

        if target_explain_group == TargetExplainGroup.CUSTOM:
            explain_targets = [0]
        else:
            explain_targets = None

        if saliency_maps.ndim == 3:
            target_explain_group = TargetExplainGroup.IMAGE
            explain_targets = None
        saliency_map_obj = ExplanationResult(
            saliency_maps, target_explain_group=target_explain_group, target_explain_indices=explain_targets
        )

        raw_sal_map_dims = len(saliency_map_obj.sal_map_shape)
        data = np.ones((20, 20, 3))
        post_processor = PostProcessor(
            saliency_map_obj,
            data,
            post_processing_parameters,
        )
        saliency_map_processed = post_processor.postprocess()

        assert saliency_map_processed is not None
        expected_dims = raw_sal_map_dims
        if colormap or overlay:
            expected_dims += 1
        assert len(saliency_map_processed.sal_map_shape) == expected_dims

        if normalize and not colormap and not overlay:
            for map_ in saliency_map_processed.saliency_map.values():
                assert map_.min() == 0, f"{map_.min()}"
                assert map_.max() in {254, 255}, f"{map_.max()}"
        if resize or overlay:
            for map_ in saliency_map_processed.saliency_map.values():
                assert map_.shape[:2] == data.shape[:2]
