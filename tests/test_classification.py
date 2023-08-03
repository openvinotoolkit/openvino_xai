import os

import numpy as np
import pytest
from urllib.request import urlretrieve

from openvino.model_api.models import ClassificationModel

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer
from openvino_xai.model import XAIClassificationModel

MODELS = [
    "mlc_mobilenetv3_large_voc",  # verified
    "mlc_efficient_b0_voc",  # verified
    "mlc_efficient_v2s_voc",  # verified
    "cls_mobilenetv3_large_cars",
    "cls_efficient_b0_cars",
    "cls_efficient_v2s_cars",
    "mobilenet_v3_large_hc_cf",
    "classification_model_with_xai_head",  # verified
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


class TestClsWB:
    _ref_sal_maps = {
        "mlc_mobilenetv3_large_voc": np.array([113, 71, 92, 101, 81, 56, 81], dtype=np.uint8),
        "mlc_efficient_b0_voc": np.array([0, 106, 177, 255, 250, 186, 6], dtype=np.uint8),
        "mlc_efficient_v2s_voc": np.array([29, 145, 195, 205, 209, 183, 52], dtype=np.uint8),
        "classification_model_with_xai_head": np.array([213, 145, 196, 245, 255, 251, 250], dtype=np.uint8),
    }

    @pytest.mark.parametrize(
        "model_name",
        MODELS,
    )
    def test_classification_white_box(self, model_name):
        data_dir = "."
        retrieve_otx_model(data_dir, model_name)
        model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
        model = XAIClassificationModel.create_model(model_path, "Classification")

        explanations = WhiteBoxExplainer(model).explain(np.zeros((224, 224, 3)))
        assert explanations is not None
        assert explanations.shape[1] == len(model.labels)
        if model_name in self._ref_sal_maps:
            actual_sal_vals = explanations[0, 0, 0, :].astype(np.int8)
            ref_sal_vals = self._ref_sal_maps[model_name].astype(np.int8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)


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
