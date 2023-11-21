from pathlib import Path

import pytest
from openvino import runtime as ov

from openvino_xai import insert_xai
from openvino_xai.common.parameters import ModelType
from openvino_xai.common.utils import retrieve_otx_model, has_xai
from tests.integration.test_classification import MODELS


@pytest.mark.parametrize("model_name", MODELS)
def test_insertion(model_name):
    data_dir = Path(".data")
    retrieve_otx_model(data_dir, model_name)
    model_path = data_dir / "otx_models" / (model_name + ".xml")

    model_ir = ov.Core().read_model(model_path)
    if model_name != "classification_model_with_xai_head":
        assert not has_xai(model_ir), "Input IR model should not have XAI head."
    if model_name == "classification_model_with_xai_head":
        assert has_xai(model_ir), "Input IR model should have XAI head."

    model_with_xai = insert_xai(model_ir, ModelType.CLASSIFICATION)

    assert has_xai(model_with_xai), "Updated IR model should has XAI head."


class TestInsertedGraph:
    # TODO: implement
    """
    Compare inserted XAI branch with a reference graph.
    Apply for a set of models
    """
