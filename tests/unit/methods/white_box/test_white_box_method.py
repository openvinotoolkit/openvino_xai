# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import openvino.runtime as ov
import pytest

from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.methods.white_box.activation_map import ActivationMap
from openvino_xai.methods.white_box.det_class_probability_map import (
    DetClassProbabilityMap,
)
from openvino_xai.methods.white_box.recipro_cam import ReciproCAM, ViTReciproCAM
from tests.intg.test_classification import DEFAULT_CLS_MODEL
from tests.intg.test_detection import DEFAULT_DET_MODEL


class TestActivationMap:
    """Test for ActivationMap."""

    _ref_sal_maps = {DEFAULT_CLS_MODEL: np.array([32, 12, 34, 47, 36, 0, 42], dtype=np.int16)}

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root: Path) -> None:
        """Setup the test case."""
        self.model_name = DEFAULT_CLS_MODEL
        retrieve_otx_model(fxt_data_root, self.model_name)
        model_path = fxt_data_root / "otx_models" / (self.model_name + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.target_layer = None

    def test_initialization(self):
        """Test ActivationMap is created properly."""
        xai_method = ActivationMap(self.model, self.target_layer, prepare_model=False)

        assert isinstance(xai_method.model_ori, ov.Model)
        assert xai_method.embed_scaling
        assert not xai_method.per_class
        assert xai_method._target_layer == self.target_layer
        assert xai_method.predictions == {}

    def test_generate_xai_branch(self):
        """Test that ActivationMap creates a proper XAI branch node."""
        activationmap_method = ActivationMap(self.model, self.target_layer, prepare_model=False)

        xai_output_node = activationmap_method.generate_xai_branch()

        # Check node's type and output shape
        assert isinstance(xai_output_node, ov.Node)
        assert len(xai_output_node.get_output_partial_shape(0)) == 3
        assert tuple(xai_output_node.get_output_partial_shape(0))[1:] == (7, 7)

        model_ori_outputs = activationmap_method.model_ori.outputs
        model_ori_params = activationmap_method.model_ori.get_parameters()
        model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

        compiled_model = ov.Core().compile_model(model_xai, "CPU")
        result = compiled_model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        raw_saliency_map = result[-1]
        assert raw_saliency_map.shape == (1, 7, 7)

        # Check that inserted XAI branch process values correctly
        actual_sal_vals = raw_saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[self.model_name]
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        # Check node's name
        xai_node_name = xai_output_node.get_friendly_name()
        assert "Multiply_" in xai_node_name

        # Check that node was inserted in the right place
        nodes_list = [op.get_friendly_name() for op in model_xai.get_ordered_ops()]
        assert nodes_list.index(xai_node_name) == 532


class TestReciproCAM:
    """Test for ReciproCAM."""

    _ref_sal_maps = {DEFAULT_CLS_MODEL: np.array([113, 71, 92, 101, 81, 56, 81], dtype=np.int16)}

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root: Path) -> None:
        """Setup the test case."""
        self.model_name = DEFAULT_CLS_MODEL
        retrieve_otx_model(fxt_data_root, self.model_name)
        model_path = fxt_data_root / "otx_models" / (self.model_name + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.target_layer = None

    def test_initialization(self):
        """Test ReciproCAM is created properly."""

        reciprocam_xai_method = ReciproCAM(self.model, self.target_layer, prepare_model=False)

        assert isinstance(reciprocam_xai_method.model_ori, ov.Model)
        assert reciprocam_xai_method.embed_scaling
        assert reciprocam_xai_method.per_class
        assert reciprocam_xai_method._target_layer == self.target_layer
        assert reciprocam_xai_method.predictions == {}

    def test_generate_xai_branch(self):
        """Test that ReciproCAM creates a proper XAI branch node."""
        reciprocam_xai_method = ReciproCAM(self.model, self.target_layer, prepare_model=False)

        xai_output_node = reciprocam_xai_method.generate_xai_branch()

        # Check node's type and output shape
        assert isinstance(xai_output_node, ov.Node)
        assert len(xai_output_node.shape) == 4
        assert tuple(xai_output_node.shape)[2:] == (7, 7)

        model_ori_outputs = reciprocam_xai_method.model_ori.outputs
        model_ori_params = reciprocam_xai_method.model_ori.get_parameters()
        model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

        compiled_model = ov.Core().compile_model(model_xai, "CPU")
        result = compiled_model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        raw_saliency_map = result[-1]
        assert raw_saliency_map.shape == (1, 20, 7, 7)

        # Check that inserted XAI branch process values correctly
        actual_sal_vals = raw_saliency_map[0][0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[self.model_name]
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        # Check node's name
        xai_node_name = xai_output_node.get_friendly_name()
        assert "Reshape_" in xai_node_name

        # Check that node was inserted in the right place
        nodes_list = [op.get_friendly_name() for op in model_xai.get_ordered_ops()]
        assert nodes_list.index(xai_node_name) == 567


class TestViTReciproCAM:
    """Test for ViTReciproCAM."""

    _ref_sal_maps = {"deit-tiny": np.array([110, 75, 47, 47, 51, 56, 62, 64, 62, 61], dtype=np.int16)}

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root: Path) -> None:
        """Setup the test case."""
        self.model_name = "deit-tiny"
        retrieve_otx_model(fxt_data_root, self.model_name)
        model_path = fxt_data_root / "otx_models" / (self.model_name + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.target_layer = None

    def test_initialization(self):
        """Test ViTReciproCAM is created properly."""

        reciprocam_xai_method = ViTReciproCAM(self.model, self.target_layer, prepare_model=False)

        assert isinstance(reciprocam_xai_method.model_ori, ov.Model)
        assert reciprocam_xai_method.embed_scaling
        assert reciprocam_xai_method.per_class
        assert reciprocam_xai_method._target_layer == self.target_layer
        assert reciprocam_xai_method.predictions == {}

    @pytest.mark.parametrize("use_gaussian", [True, False])
    def test_generate_xai_branch(self, use_gaussian):
        """Test that ViTReciproCAM creates a proper XAI branch node."""
        reciprocam_xai_method = ViTReciproCAM(
            self.model, self.target_layer, use_gaussian=use_gaussian, prepare_model=False
        )

        xai_output_node = reciprocam_xai_method.generate_xai_branch()

        # Check node's type and output shape
        assert isinstance(xai_output_node, ov.Node)
        assert len(xai_output_node.shape) == 4
        assert tuple(xai_output_node.shape)[2:] == (14, 14)

        model_ori_outputs = reciprocam_xai_method.model_ori.outputs
        model_ori_params = reciprocam_xai_method.model_ori.get_parameters()
        model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

        compiled_model = ov.Core().compile_model(model_xai, "CPU")
        result = compiled_model(np.zeros((1, 3, 224, 224), dtype=np.float32))
        raw_saliency_map = result[-1]
        assert raw_saliency_map.shape == (1, 10, 14, 14)

        # Check that inserted XAI branch process values correctly
        if use_gaussian:
            actual_sal_vals = raw_saliency_map[0][0][0, :10].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps[self.model_name]
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        # Check node's name
        xai_node_name = xai_output_node.get_friendly_name()
        assert "Reshape_" in xai_node_name

        # Check that node was inserted in the right place
        if use_gaussian:
            nodes_list = [op.get_friendly_name() for op in model_xai.get_ordered_ops()]
            assert nodes_list.index(xai_node_name) == 856


class TestDetProbMapXAI:
    """Tests for DetClassProbabilityMap."""

    # TODO: Support check xai_branch == xai_branch_reference
    # TODO: Create small model to use it as mocker model
    # TODO: Add check that model with XAI branch is equal to reference graph (DFS)
    # TODO: Add insertion node check (target node is found correctly in auto mode)
    _ref_sal_maps = {
        "det_mobilenetv2_atss_bccd": np.array([234, 203, 190, 196, 208, 206, 201, 199, 192, 186], dtype=np.uint8)
    }

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root: Path) -> None:
        """Setup the test case."""
        retrieve_otx_model(fxt_data_root, DEFAULT_DET_MODEL)
        model_path = fxt_data_root / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.target_layer = [
            "/bbox_head/atss_cls_1/Conv/WithoutBiases",
            "/bbox_head/atss_cls_2/Conv/WithoutBiases",
            "/bbox_head/atss_cls_3/Conv/WithoutBiases",
            "/bbox_head/atss_cls_4/Conv/WithoutBiases",
        ]
        self.num_anchors = [1, 1, 1, 1, 1]

    def test_initialization(self):
        """Test DetClassProbabilityMap is created properly."""

        detection_xai_method = DetClassProbabilityMap(
            self.model,
            target_layer=self.target_layer,
            num_anchors=self.num_anchors,
            prepare_model=False,
        )

        assert isinstance(detection_xai_method.model_ori, ov.Model)
        assert detection_xai_method.embed_scaling
        assert detection_xai_method.per_class
        assert detection_xai_method._target_layer == self.target_layer
        assert detection_xai_method._num_anchors == self.num_anchors
        assert detection_xai_method._saliency_map_size == (23, 23)
        assert detection_xai_method.predictions == {}

    def test_generate_xai_branch(self):
        """Test that DetClassProbabilityMap creates a proper XAI branch node."""
        detection_xai_method = DetClassProbabilityMap(
            self.model,
            target_layer=self.target_layer,
            num_anchors=self.num_anchors,
            prepare_model=False,
        )

        xai_output_node = detection_xai_method.generate_xai_branch()

        # Check node's type and output shape
        assert isinstance(xai_output_node, ov.Node)
        assert len(xai_output_node.shape) == 4
        assert tuple(xai_output_node.shape)[2:] == (23, 23)

        model_ori_outputs = detection_xai_method.model_ori.outputs
        model_ori_params = detection_xai_method.model_ori.get_parameters()
        model_xai = ov.Model([*model_ori_outputs, xai_output_node.output(0)], model_ori_params)

        compiled_model = ov.Core().compile_model(model_xai, "CPU")
        result = compiled_model(np.zeros((1, 3, 736, 992), dtype=np.float32))
        raw_saliency_map = result[-1]
        assert raw_saliency_map.shape == (1, 3, 23, 23)

        # Check that inserted XAI branch process values correctly
        actual_sal_vals = raw_saliency_map[0][0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[DEFAULT_DET_MODEL].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        # Check node's name
        xai_node_name = xai_output_node.get_friendly_name()
        assert "Reshape_" in xai_node_name

        # Check that node was inserted in the right place
        nodes_list = [op.get_friendly_name() for op in model_xai.get_ordered_ops()]
        assert nodes_list.index(xai_node_name) == 551
