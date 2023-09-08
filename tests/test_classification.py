import os
from pathlib import Path

import numpy as np
import pytest
from urllib.request import urlretrieve

from openvino.model_api.models import ClassificationModel
import openvino.runtime as ov

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer
from openvino_xai.parameters import ClassificationExplainParametersWB, PostProcessParameters, XAIMethodType
from openvino_xai.saliency_map import TargetExplainGroup
from openvino_xai.model import XAIClassificationModel, XAIModel


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

DEFAULT_MODEL = "mlc_efficient_b0_voc"

MODELS_NUM_CLASSES = {
    "mlc_mobilenetv3_large_voc": 20,  # verified
    "mlc_efficient_b0_voc": 20,  # verified
    "mlc_efficient_v2s_voc": 20,  # verified
    "cls_mobilenetv3_large_cars": 196,
    "cls_efficient_b0_cars": 196,
    "cls_efficient_v2s_cars": 196,
    "mobilenet_v3_large_hc_cf": 8,
    "classification_model_with_xai_head": 4,  # verified
}


def retrieve_otx_model(data_dir, model_name):
    destination_folder = Path(data_dir) / "otx_models"
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

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL_CLASSES,
            TargetExplainGroup.CUSTOM_CLASSES,
        ],
    )
    def test_reciprocam(self, model_name, embed_normalization, target_explain_group):
        data_dir = ".data"
        retrieve_otx_model(data_dir, model_name)
        model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
        explain_parameters = ClassificationExplainParametersWB(
            embed_normalization=embed_normalization,
            explain_method_type=XAIMethodType.RECIPROCAM,
        )
        model = XAIClassificationModel.create_model(
            model_path, "Classification", explain_parameters=explain_parameters
        )

        if target_explain_group == TargetExplainGroup.ALL_CLASSES:
            explanations = WhiteBoxExplainer(model).explain(np.zeros((224, 224, 3)), target_explain_group)
            assert explanations is not None
            assert explanations.map.shape[1] == len(model.labels)
            if model_name in self._ref_sal_maps:
                actual_sal_vals = explanations.map[0, 0, 0, :].astype(np.int8)
                ref_sal_vals = self._ref_sal_maps[model_name].astype(np.int8)
                if embed_normalization:
                    # Reference values generated with embed_normalization=True
                    assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
                else:
                    if model_name == "classification_model_with_xai_head":
                        pytest.skip("model already has xai head - this test cannot change it.")
                    assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100
        if target_explain_group == TargetExplainGroup.CUSTOM_CLASSES:
            explanations = WhiteBoxExplainer(model).explain(
                np.zeros((224, 224, 3)), target_explain_group, [0]
            )
            assert explanations is not None
            assert explanations.map.ndim == 3

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    def test_activationmap(self, model_name, embed_normalization):
        if model_name == "classification_model_with_xai_head":
            pytest.skip("model already has reciprocam xai head - this test cannot change it.")
        data_dir = ".data"
        retrieve_otx_model(data_dir, model_name)
        model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
        explain_parameters = ClassificationExplainParametersWB(
            embed_normalization=embed_normalization,
            explain_method_type=XAIMethodType.ACTIVATIONMAP,
        )
        model = XAIClassificationModel.create_model(
            model_path, "Classification", explain_parameters=explain_parameters
        )

        explanations = WhiteBoxExplainer(model).explain(np.zeros((224, 224, 3)))
        assert explanations is not None
        assert explanations.map.ndim == 3

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("explain_method_type", [XAIMethodType.RECIPROCAM, XAIMethodType.ACTIVATIONMAP])
    @pytest.mark.parametrize("overlay", [True, False])
    def test_classification_white_box_postprocessing(self, model_name, explain_method_type, overlay):
        data_dir = ".data"
        retrieve_otx_model(data_dir, model_name)
        model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
        explain_parameters = ClassificationExplainParametersWB(
            explain_method_type=explain_method_type,
        )
        model = XAIClassificationModel.create_model(
            model_path, "Classification", explain_parameters=explain_parameters
        )

        target_explain_group = None
        if model_name == "classification_model_with_xai_head":
            target_explain_group = TargetExplainGroup.ALL_CLASSES

        post_processing_parameters = PostProcessParameters(overlay=overlay)
        explanations = WhiteBoxExplainer(model).explain(
            np.zeros((224, 224, 3)),
            target_explain_group=target_explain_group,
            post_processing_parameters=post_processing_parameters,
        )
        assert explanations is not None
        if explain_method_type == XAIMethodType.RECIPROCAM:
            if overlay:
                assert explanations.map.shape == (1, MODELS_NUM_CLASSES[model_name], 224, 224, 3)
            else:
                assert explanations.map.shape == (1, MODELS_NUM_CLASSES[model_name], 7, 7)
        if explain_method_type == XAIMethodType.ACTIVATIONMAP:
            if model_name == "classification_model_with_xai_head":
                pytest.skip("model already has xai head - this test cannot change it.")
            if overlay:
                assert explanations.map.shape == (1, 224, 224, 3)
            else:
                assert explanations.map.shape == (1, 7, 7)


@pytest.mark.parametrize("model_name", MODELS)
def test_classification_auto(model_name):
    data_dir = ".data"
    retrieve_otx_model(data_dir, model_name)
    model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
    model = ClassificationModel.create_model(model_path, "Classification")
    target_explain_group = None
    if model_name == "classification_model_with_xai_head":
        target_explain_group = TargetExplainGroup.ALL_CLASSES
    explanations = ClassificationAutoExplainer(model).explain(np.zeros((224, 224, 3)), target_explain_group)
    assert explanations is not None


@pytest.mark.parametrize("model_name", MODELS)
def test_ir_model_update_wo_inference(model_name):
    data_dir = ".data"
    retrieve_otx_model(data_dir, model_name)
    model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")

    model_ir = ov.Core().read_model(model_path)
    if model_name != "classification_model_with_xai_head":
        assert not XAIModel.has_xai(model_ir), "Input IR model should not have XAI head."
    if model_name == "classification_model_with_xai_head":
        assert XAIModel.has_xai(model_ir), "Input IR model should have XAI head."

    output = os.path.join(data_dir, "otx_models")
    model_with_xai = XAIClassificationModel.insert_xai_into_native_ir(model_path, output)

    assert XAIModel.has_xai(model_with_xai), "Updated IR model should has XAI head."
    model_name = Path(model_path).stem
    if model_name != "classification_model_with_xai_head":
        assert os.path.exists(
            os.path.join(output, model_name + "_xai.xml")
        ), "Updated IR model should be saved."


def test_classification_explain_parameters():
    cls_explain_params = ClassificationExplainParametersWB()
    assert cls_explain_params.target_layer is None
    assert cls_explain_params.embed_normalization
    assert cls_explain_params.explain_method_type == XAIMethodType.RECIPROCAM
