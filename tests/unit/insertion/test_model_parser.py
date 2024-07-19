# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
import pytest

from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.inserter.model_parser import IRParserCls, ModelType
from tests.intg.test_classification import DEFAULT_CLS_MODEL


class TestIRParser:
    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    def test_get_logit_node(self):
        model = self._get_model()
        logit_node = IRParserCls.get_logit_node(model)
        assert "Multiply_" in logit_node.name

    def test_get_post_target_node(self):
        model = self._get_model()
        post_target_node = IRParserCls.get_post_target_node(model, model_type=ModelType.CNN)
        assert "ReduceMean_" in post_target_node[0].name

    def test_get_first_conv_node(self):
        model = self._get_model()
        first_conv_node = IRParserCls.get_first_conv_node(model)
        assert "Convolution_" in first_conv_node.name

    def _get_model(self):
        retrieve_otx_model(self.data_dir, DEFAULT_CLS_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        return ov.Core().read_model(model_path)
