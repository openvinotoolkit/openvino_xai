# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino import runtime as ov

from openvino_xai.api.api import insert_xai
from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import has_xai, retrieve_otx_model
from openvino_xai.inserter.parameters import DetectionInsertionParameters
from tests.integration.test_classification import DATA_DIR, MODELS
from tests.integration.test_detection import DEFAULT_DET_MODEL, MODEL_CONFIGS


@pytest.mark.parametrize("model_name", MODELS)
def test_insertion_classification(model_name):
    retrieve_otx_model(DATA_DIR, model_name)
    model_path = DATA_DIR / "otx_models" / (model_name + ".xml")

    model_ir = ov.Core().read_model(model_path)
    if model_name != "classification_model_with_xai_head":
        assert not has_xai(model_ir), "Input IR model should not have XAI head."
    if model_name == "classification_model_with_xai_head":
        assert has_xai(model_ir), "Input IR model should have XAI head."

    model_with_xai = insert_xai(model_ir, Task.CLASSIFICATION)

    assert has_xai(model_with_xai), "Updated IR model should has XAI head."


def test_insertion_detection():
    retrieve_otx_model(DATA_DIR, DEFAULT_DET_MODEL)
    model_path = DATA_DIR / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
    model = ov.Core().read_model(model_path)

    cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names
    insertion_parameters = DetectionInsertionParameters(
        target_layer=cls_head_output_node_names,
        num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
        explain_method=Method.DETCLASSPROBABILITYMAP,
    )

    model_with_xai = insert_xai(model, Task.DETECTION, insertion_parameters)
    assert has_xai(model_with_xai), "Updated IR model should has XAI head."


class TestInsertedGraph:
    # TODO: implement
    """
    Compare inserted XAI branch with a reference graph.
    Apply for a set of models
    """
