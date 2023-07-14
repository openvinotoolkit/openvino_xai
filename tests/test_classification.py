import os

import numpy as np
import pytest
from urllib.request import urlretrieve

from openvino.model_api.models import ClassificationModel

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer
from openvino_xai.model import XAIClassificationModel

MODELS = [
    "mlc_mobilenetv3_large_voc",
    "mlc_efficient_b0_voc",
    "mlc_efficient_v2s_voc",
    "cls_mobilenetv3_large_cars",
    "cls_efficient_b0_cars",
    "cls_efficient_v2s_cars",
    "mobilenet_v3_large_hc_cf",
    "classification_model_with_xai_head",
]


def retrieve_otx_model(data_dir, model_name):
    destination_folder = os.path.join(data_dir, "otx_models")
    os.makedirs(destination_folder, exist_ok=True)

    if not os.path.isfile(os.path.join(destination_folder, model_name + ".xml")):
        urlretrieve(
            f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.xml",
            f"{destination_folder}/{model_name}.xml",
        )
    if not os.path.isfile(os.path.join(destination_folder, model_name + ".bin")):
        urlretrieve(
            f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.bin",
            f"{destination_folder}/{model_name}.bin",
        )


@pytest.mark.parametrize(
    "model_name",
    MODELS,
)
def test_classification_white_box(model_name):
    data_dir = "."
    retrieve_otx_model(data_dir, model_name)
    model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
    model = XAIClassificationModel.create_model(model_path, "Classification")
    explanations = WhiteBoxExplainer(model).explain(np.zeros((224, 224, 3)))
    assert explanations is not None


@pytest.mark.parametrize(
    "model_name",
    MODELS,
)
def test_classification_auto(model_name):
    data_dir = "."
    retrieve_otx_model(data_dir, model_name)
    model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
    model = ClassificationModel.create_model(model_path, "Classification")
    explanations = ClassificationAutoExplainer(model).explain(np.zeros((224, 224, 3)))
    assert explanations is not None
