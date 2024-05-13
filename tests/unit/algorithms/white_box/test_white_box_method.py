# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import openvino.runtime as ov
import pytest

from openvino_xai.algorithms.white_box.white_box_methods import (
    DetClassProbabilityMapXAIMethod,
)
from openvino_xai.common.utils import retrieve_otx_model


class TestDetProbMapXAI:
    """Tests for DetClassProbabilityMapXAIMethod."""

    # TODO: Support check xai_branch == xai_branch_reference
    # TODO: Create small model to use it as mocker model
    # TODO: Add check that model with XAI branch is equal to reference graph (DFS)
    # TODO: Add insertion node check (target node is found correctly in auto mode)
    _ref_sal_maps = {
        "det_mobilenetv2_atss_bccd": np.array([234, 203, 190, 196, 208, 206, 201, 199, 192, 186], dtype=np.uint8)
    }

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """Setup the test case."""
        data_dir = Path(".data")
        self.model_name = "det_mobilenetv2_atss_bccd"
        retrieve_otx_model(data_dir, self.model_name)
        model_path = data_dir / "otx_models" / (self.model_name + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.target_layer = [
            "/bbox_head/atss_cls_1/Conv/WithoutBiases",
            "/bbox_head/atss_cls_2/Conv/WithoutBiases",
            "/bbox_head/atss_cls_3/Conv/WithoutBiases",
            "/bbox_head/atss_cls_4/Conv/WithoutBiases",
        ]
        self.num_anchors = [1, 1, 1, 1, 1]

    def test_initialization(self):
        """Test DetClassProbabilityMapXAIMethod is created properly."""

        detection_xai_method = DetClassProbabilityMapXAIMethod(self.model, self.target_layer, self.num_anchors)

        assert detection_xai_method.model_ori == self.model
        assert isinstance(detection_xai_method.model_ori, ov.Model)
        assert detection_xai_method.embed_normalization
        assert detection_xai_method.per_class
        assert len(detection_xai_method.supported_target_explain_groups) == 2
        assert detection_xai_method._target_layer == self.target_layer
        assert detection_xai_method._num_anchors == self.num_anchors
        assert detection_xai_method._saliency_map_size == (23, 23)

    def test_generate_xai_branch(self):
        """Test that DetClassProbabilityMapXAIMethod creates a proper XAI branch node."""
        detection_xai_method = DetClassProbabilityMapXAIMethod(self.model, self.target_layer, self.num_anchors)

        xai_output_node = detection_xai_method.generate_xai_branch()

        # Check node's type and output shape
        assert isinstance(xai_output_node, ov.Node)
        assert len(xai_output_node.shape) == 4
        assert tuple(xai_output_node.shape)[2:] == (23, 23)

        model_ori_outputs = self.model.outputs
        model_ori_params = self.model.get_parameters()
        model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

        compiled_model = ov.Core().compile_model(model_xai, "CPU")
        result = compiled_model(np.zeros((1, 3, 736, 992), dtype=np.float32))
        raw_saliency_map = result[-1]
        assert raw_saliency_map.shape == (1, 3, 23, 23)

        # Check that inserted XAI branch process values correctly
        actual_sal_vals = raw_saliency_map[0][0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[self.model_name].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        # Check node's name
        xai_node_name = xai_output_node.get_friendly_name()
        assert "Reshape_" in xai_node_name

        # Check that node was inserted in the right place
        nodes_list = [op.get_friendly_name() for op in model_xai.get_ordered_ops()]
        assert nodes_list.index(xai_node_name) == 551
