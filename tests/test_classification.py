import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from urllib.request import urlretrieve

from openvino.model_api.models import ClassificationModel
import openvino.runtime as ov

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer, RISEExplainer
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

DEFAULT_MODEL = "mlc_mobilenetv3_large_voc"

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
    image = cv2.imread("tests/assets/cheetah_class293.jpg")
    _ref_sal_maps_reciprocam = {
        "mlc_mobilenetv3_large_voc": np.array([215, 214, 233, 239, 218, 206, 210], dtype=np.uint8),
        "mlc_efficient_b0_voc": np.array([73, 242, 156, 219, 197, 239, 69], dtype=np.uint8),
        "mlc_efficient_v2s_voc": np.array([157, 166, 166, 181, 162, 151, 147], dtype=np.uint8),
        "classification_model_with_xai_head": np.array([172, 173, 235, 236, 237, 238, 227], dtype=np.uint8),
    }
    _ref_sal_maps_activationmap = {
        "mlc_mobilenetv3_large_voc": np.array([7, 7, 13, 16, 3, 0, 5], dtype=np.uint8),
    }

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL_CLASSES,
            TargetExplainGroup.PREDICTED_CLASSES,
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
            explanations = WhiteBoxExplainer(model).explain(self.image, target_explain_group)
            assert explanations is not None
            assert len(explanations.map) == len(model.labels)
            if model_name in self._ref_sal_maps_reciprocam:
                actual_sal_vals = explanations.map[0][0, :].astype(np.int16)
                ref_sal_vals = self._ref_sal_maps_reciprocam[model_name].astype(np.uint8)
                if embed_normalization:
                    # Reference values generated with embed_normalization=True
                    assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
                else:
                    if model_name == "classification_model_with_xai_head":
                        pytest.skip("model already has fixed xai head - this test cannot change it.")
                    assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100
        if target_explain_group == TargetExplainGroup.PREDICTED_CLASSES:
            explanations = WhiteBoxExplainer(model).explain(self.image, target_explain_group)
            assert explanations is not None
            assert len(explanations.map) == len(explanations.predictions)
        if target_explain_group == TargetExplainGroup.CUSTOM_CLASSES:
            target_class = 1
            explanations = WhiteBoxExplainer(model).explain(
                self.image, target_explain_group, [target_class]
            )
            assert explanations is not None
            assert target_class in explanations.map
            assert len(explanations.map) == len([target_class])
            assert explanations.map[target_class].ndim == 2

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

        explanations = WhiteBoxExplainer(model).explain(self.image)
        if model_name in self._ref_sal_maps_activationmap and embed_normalization:
            actual_sal_vals = explanations.map["per_image_map"][0, :].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps_activationmap[model_name].astype(np.uint8)
            # Reference values generated with embed_normalization=True
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        assert explanations is not None
        assert "per_image_map" in explanations.map
        assert explanations.map["per_image_map"].ndim == 2

    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL_CLASSES,
            TargetExplainGroup.PREDICTED_CLASSES,
            TargetExplainGroup.CUSTOM_CLASSES,
        ],
    )
    @pytest.mark.parametrize("overlay", [True, False])
    def test_classification_postprocessing(self, target_explain_group, overlay):
        data_dir = ".data"
        retrieve_otx_model(data_dir, DEFAULT_MODEL)
        model_path = os.path.join(data_dir, "otx_models", DEFAULT_MODEL + ".xml")
        model = XAIClassificationModel.create_model(model_path, "Classification")

        explain_targets = None
        if target_explain_group == TargetExplainGroup.CUSTOM_CLASSES:
            explain_targets = [0]
        post_processing_parameters = PostProcessParameters(overlay=overlay)
        explanations = WhiteBoxExplainer(model).explain(
            self.image,
            target_explain_group=target_explain_group,
            explain_targets=explain_targets,
            post_processing_parameters=post_processing_parameters,
        )
        assert explanations is not None
        if target_explain_group == TargetExplainGroup.ALL_CLASSES:
            assert len(explanations.map) == MODELS_NUM_CLASSES[DEFAULT_MODEL]
        if target_explain_group == TargetExplainGroup.PREDICTED_CLASSES:
            assert len(explanations.map) == len(explanations.predictions)
        if target_explain_group == TargetExplainGroup.CUSTOM_CLASSES:
            assert len(explanations.map) == len(explain_targets)
        if overlay:
            assert explanations.sal_map_shape == (354, 500, 3)
        else:
            assert explanations.sal_map_shape == (7, 7)
            for map_ in explanations.map.values():
                assert map_.min() == 0, f"{map_.min()}"
                assert map_.max() in {254, 255}, f"{map_.max()}"

    def test_two_sequential_norms(self):
        data_dir = ".data"
        retrieve_otx_model(data_dir, DEFAULT_MODEL)
        model_path = os.path.join(data_dir, "otx_models", DEFAULT_MODEL + ".xml")
        model = XAIClassificationModel.create_model(
            model_path, "Classification", explain_parameters=ClassificationExplainParametersWB(embed_normalization=True)
        )
        explanations = WhiteBoxExplainer(model).explain(
            self.image,
            target_explain_group=TargetExplainGroup.ALL_CLASSES,
            post_processing_parameters=PostProcessParameters(normalize=True),
        )

        actual_sal_vals = explanations.map[0][0, :].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps_reciprocam[DEFAULT_MODEL].astype(np.uint8)
        # Reference values generated with embed_normalization=True
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        for map_ in explanations.map.values():
            assert map_.min() == 0, f"{map_.min()}"
            assert map_.max() in {254, 255}, f"{map_.max()}"


@pytest.mark.parametrize("model_name", MODELS)
def test_classification_auto(model_name):
    # TODO provide incorrect explain params so that WB fails and BB will work
    data_dir = ".data"
    retrieve_otx_model(data_dir, model_name)
    model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")
    model = ClassificationModel.create_model(model_path, "Classification")
    target_explain_group = None
    if model_name == "classification_model_with_xai_head":
        target_explain_group = TargetExplainGroup.ALL_CLASSES
    explanations = ClassificationAutoExplainer(model).explain(
        cv2.imread("tests/assets/cheetah_class293.jpg"), target_explain_group
    )
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


class TestClsBB:
    image = cv2.imread("tests/assets/cheetah_class293.jpg")
    _ref_sal_maps = {
        "mlc_mobilenetv3_large_voc": np.array([21, 25, 30, 34, 38, 42, 47, 51, 57, 64], dtype=np.uint8),
        "mlc_efficient_b0_voc": np.array([13, 17, 20, 23, 27, 30, 33, 37, 42, 49], dtype=np.uint8),
        "mlc_efficient_v2s_voc": np.array([20, 24, 28, 32, 36, 40, 44, 48, 54, 61], dtype=np.uint8),
        "classification_model_with_xai_head": np.array([15, 18, 22, 26, 29, 33, 37, 40, 46, 53], dtype=np.uint8),
    }

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("overlay", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL_CLASSES,
            TargetExplainGroup.CUSTOM_CLASSES,
        ],
    )
    def test_classification_black_box_postprocessing(self, model_name, overlay, target_explain_group):
        data_dir = ".data"
        retrieve_otx_model(data_dir, model_name)
        model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")

        model = ClassificationModel.create_model(
        model_path, model_type="Classification", configuration={"output_raw_scores": True}
        )
        explainer = RISEExplainer(model, num_masks=5)
        post_processing_parameters = PostProcessParameters(
            overlay=overlay,
        )
        if target_explain_group == TargetExplainGroup.CUSTOM_CLASSES:
            target_class = 1
            explanation = explainer.explain(
                self.image,
                target_explain_group,
                [target_class]
            )
            assert explanation is not None
            assert target_class in explanation.map
            assert len(explanation.map) == len([target_class])
            assert explanation.map[target_class].ndim == 2
        else:
            explanation = explainer.explain(
                self.image,
                target_explain_group,
                post_processing_parameters=post_processing_parameters,
            )
            assert explanation is not None
            if overlay:
                assert len(explanation.map) == MODELS_NUM_CLASSES[model_name]
                assert explanation.sal_map_shape == (354, 500, 3)
            else:
                assert len(explanation.map) == MODELS_NUM_CLASSES[model_name]
                assert explanation.sal_map_shape == (224, 224)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_classification_black_box_pred_class(self, model_name):
        data_dir = ".data"
        retrieve_otx_model(data_dir, model_name)
        model_path = os.path.join(data_dir, "otx_models", model_name + ".xml")

        model = ClassificationModel.create_model(
        model_path, model_type="Classification", configuration={"output_raw_scores": True}
        )
        explainer = RISEExplainer(model, num_masks=5)

        explanation = explainer.explain(
            self.image,
            TargetExplainGroup.PREDICTED_CLASSES)
        assert explanation is not None
        assert len(explanation.map) > 0
        assert len(explanation.map) == len(explanation.predictions)
        assert explanation.sal_map_shape == (224, 224)

        # Check that returned saliency map classes and predicted classes are the same
        predicted_class_idx = sorted([pred[0] for pred in explanation.predictions])
        returned_sal_map_classes = list(sorted(explanation.map.keys()))
        assert predicted_class_idx == returned_sal_map_classes

        if model_name in self._ref_sal_maps:
            first_idx = predicted_class_idx[0]
            actual_sal_vals = explanation.map[first_idx][0, :10].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps[model_name].astype(np.uint8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_classification_black_box_xai_model_as_input(self):
        data_dir = ".data"
        retrieve_otx_model(data_dir, DEFAULT_MODEL)
        model_path = os.path.join(data_dir, "otx_models", DEFAULT_MODEL + ".xml")

        model = XAIClassificationModel.create_model(
            model_path, model_type="Classification", configuration={"output_raw_scores": True}
        )
        assert XAIModel.has_xai(model.inference_adapter.model), "Updated IR model should has XAI head."
        explainer = RISEExplainer(model, num_masks=5)
        explanation = explainer.explain(self.image)

        predicted_class_idx = sorted([pred[0] for pred in explanation.predictions])
        first_idx = predicted_class_idx[0]
        actual_sal_vals = explanation.map[first_idx][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[DEFAULT_MODEL].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
