# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
import pytest

import openvino_xai.api.api as xai
from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import has_xai, retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer
from openvino_xai.explainer.parameters import (
    ExplainMode,
    ExplanationParameters,
    TargetExplainGroup,
    VisualizationParameters,
)
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.inserter.parameters import ClassificationInsertionParameters

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


MODEL_NUM_CLASSES = {
    "mlc_mobilenetv3_large_voc": 20,
    "mlc_efficient_b0_voc": 20,
    "mlc_efficient_v2s_voc": 20,
    "cls_mobilenetv3_large_cars": 196,
    "cls_efficient_b0_cars": 196,
    "cls_efficient_v2s_cars": 196,
    "mobilenet_v3_large_hc_cf": 8,
    "classification_model_with_xai_head": 4,
    "deit-tiny": 10,
}


MODELS_VOC = [
    "mlc_mobilenetv3_large_voc",  # verified
    "mlc_efficient_b0_voc",  # verified
    "mlc_efficient_v2s_voc",  # verified
    "mobilenet_v3_large_hc_cf",
]


DEFAULT_CLS_MODEL = "mlc_mobilenetv3_large_voc"
DATA_DIR = Path(".data")


class TestClsWB:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    data_dir = Path(".data")
    _ref_sal_maps_reciprocam = {
        "mlc_mobilenetv3_large_voc": np.array([236, 237, 244, 252, 242, 225, 231], dtype=np.uint8),
        "mlc_efficient_b0_voc": np.array([53, 128, 70, 234, 227, 255, 59], dtype=np.uint8),
        "mlc_efficient_v2s_voc": np.array([144, 105, 116, 195, 209, 176, 176], dtype=np.uint8),
        "classification_model_with_xai_head": np.array([165, 161, 209, 211, 208, 206, 196], dtype=np.uint8),
    }
    _ref_sal_maps_vitreciprocam = {
        "deit-tiny": np.array([200, 171, 183, 196, 198, 196, 205, 225, 207, 173, 174, 134, 97, 117], dtype=np.uint8)
    }
    _ref_sal_maps_activationmap = {
        "mlc_mobilenetv3_large_voc": np.array([6, 3, 10, 15, 5, 0, 13], dtype=np.uint8),
    }
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )

    @pytest.mark.parametrize("embed_scaling", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.CUSTOM,
        ],
    )
    def test_vitreciprocam(self, embed_scaling: bool, target_explain_group: TargetExplainGroup | TargetExplainGroup):
        model_name = "deit-tiny"
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")

        model = ov.Core().read_model(model_path)
        insertion_parameters = ClassificationInsertionParameters(
            embed_scaling=embed_scaling,
            explain_method=Method.VITRECIPROCAM,
        )

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,  # type: ignore
            explain_mode=ExplainMode.WHITEBOX,
            insertion_parameters=insertion_parameters,
        )

        if target_explain_group == TargetExplainGroup.ALL:
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                visualization_parameters=VisualizationParameters(),
            )
            explanation = explainer(self.image, explanation_parameters)
            assert explanation is not None
            assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[model_name]
            if model_name in self._ref_sal_maps_vitreciprocam:
                actual_sal_vals = explanation.saliency_map[0][0, :].astype(np.int16)
                ref_sal_vals = self._ref_sal_maps_vitreciprocam[model_name].astype(np.uint8)
                if embed_scaling:
                    # Reference values generated with embed_scaling=True
                    assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
                else:
                    assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100

        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = 1
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                target_explain_labels=[target_class],
                visualization_parameters=VisualizationParameters(),
            )
            explanation = explainer(self.image, explanation_parameters)
            assert explanation is not None
            assert target_class in explanation.saliency_map
            assert len(explanation.saliency_map) == len([target_class])
            assert explanation.saliency_map[target_class].ndim == 2

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_scaling", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.CUSTOM,
        ],
    )
    def test_reciprocam(
        self, model_name: str, embed_scaling: bool, target_explain_group: TargetExplainGroup | TargetExplainGroup
    ):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)
        insertion_parameters = ClassificationInsertionParameters(
            embed_scaling=embed_scaling,
            explain_method=Method.RECIPROCAM,
        )

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,  # type: ignore
            explain_mode=ExplainMode.WHITEBOX,
            insertion_parameters=insertion_parameters,
        )

        if target_explain_group == TargetExplainGroup.ALL:
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                visualization_parameters=VisualizationParameters(),
            )
            explanation = explainer(self.image, explanation_parameters)
            assert explanation is not None
            assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[model_name]
            if model_name in self._ref_sal_maps_reciprocam:
                actual_sal_vals = explanation.saliency_map[0][0, :].astype(np.int16)
                ref_sal_vals = self._ref_sal_maps_reciprocam[model_name].astype(np.uint8)
                if embed_scaling:
                    # Reference values generated with embed_scaling=True
                    assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
                else:
                    if model_name == "classification_model_with_xai_head":
                        pytest.skip("model already has fixed xai head - this test cannot change it.")
                    assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100

        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = 1
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                target_explain_labels=[target_class],
                visualization_parameters=VisualizationParameters(),
            )
            explanation = explainer(self.image, explanation_parameters)
            assert explanation is not None
            assert target_class in explanation.saliency_map
            assert len(explanation.saliency_map) == len([target_class])
            assert explanation.saliency_map[target_class].ndim == 2

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_scaling", [True, False])
    def test_activationmap(self, model_name: str, embed_scaling: bool):
        if model_name == "classification_model_with_xai_head":
            pytest.skip("model already has reciprocam xai head - this test cannot change it.")
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)
        insertion_parameters = ClassificationInsertionParameters(
            embed_scaling=embed_scaling,
            explain_method=Method.ACTIVATIONMAP,
        )

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,  # type: ignore
            explain_mode=ExplainMode.WHITEBOX,
            insertion_parameters=insertion_parameters,
        )

        explanation_parameters = ExplanationParameters(
            visualization_parameters=VisualizationParameters(),
        )
        explanation = explainer(self.image, explanation_parameters)
        if model_name in self._ref_sal_maps_activationmap and embed_scaling:
            actual_sal_vals = explanation.saliency_map["per_image_map"][0, :].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps_activationmap[model_name].astype(np.uint8)
            # Reference values generated with embed_scaling=True
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        assert explanation is not None
        assert "per_image_map" in explanation.saliency_map
        assert explanation.saliency_map["per_image_map"].ndim == 2

    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.CUSTOM,
        ],
    )
    @pytest.mark.parametrize("overlay", [True, False])
    def test_classification_postprocessing(
        self, target_explain_group: TargetExplainGroup | TargetExplainGroup, overlay: bool
    ):
        retrieve_otx_model(self.data_dir, DEFAULT_CLS_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,  # type: ignore
            explain_mode=ExplainMode.WHITEBOX,
        )

        explain_targets = None
        if target_explain_group == TargetExplainGroup.CUSTOM:
            explain_targets = [1]
        visualization_parameters = VisualizationParameters(overlay=overlay)

        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            target_explain_labels=explain_targets,  # type: ignore
            visualization_parameters=visualization_parameters,
        )
        explanation = explainer(self.image, explanation_parameters)
        assert explanation is not None
        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[DEFAULT_CLS_MODEL]
        if target_explain_group == TargetExplainGroup.CUSTOM:
            assert len(explanation.saliency_map) == len(explain_targets)
            assert 1 in explanation.saliency_map
        if overlay:
            assert explanation.shape == (354, 500, 3)
        else:
            assert explanation.shape == (7, 7)
            for map_ in explanation.saliency_map.values():
                assert map_.min() == 0, f"{map_.min()}"
                assert map_.max() in {254, 255}, f"{map_.max()}"

    def test_two_sequential_norms(self):
        retrieve_otx_model(self.data_dir, DEFAULT_CLS_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.ALL,
            visualization_parameters=VisualizationParameters(scaling=True),
        )
        explanation = explainer(self.image, explanation_parameters)

        actual_sal_vals = explanation.saliency_map[0][0, :].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps_reciprocam[DEFAULT_CLS_MODEL].astype(np.uint8)
        # Reference values generated with embed_scaling=True
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        for map_ in explanation.saliency_map.values():
            assert map_.min() == 0, f"{map_.min()}"
            assert map_.max() in {254, 255}, f"{map_.max()}"


class TestClsBB:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    data_dir = Path(".data")
    _ref_sal_maps = {
        "mlc_mobilenetv3_large_voc": np.array([246, 241, 236, 231, 226, 221, 216, 211, 205, 197], dtype=np.uint8),
    }
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("overlay", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.CUSTOM,
        ],
    )
    @pytest.mark.parametrize("scaling", [True, False])
    def test_classification_black_box_postprocessing(
        self,
        model_name: str,
        overlay: bool,
        target_explain_group: TargetExplainGroup | TargetExplainGroup,
        scaling: bool,
    ):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,  # type: ignore
            postprocess_fn=get_postprocess_fn(),  # type: ignore
            explain_mode=ExplainMode.BLACKBOX,
        )

        visualization_parameters = VisualizationParameters(
            overlay=overlay,
        )

        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = 1
            explanation_parameters = ExplanationParameters(
                visualization_parameters=visualization_parameters,
                target_explain_group=target_explain_group,
                target_explain_labels=[target_class],
            )

            explanation = explainer(
                self.image,
                explanation_parameters,
                num_masks=5,
                scale_output=scaling,
            )

            assert explanation is not None
            assert target_class in explanation.saliency_map
            assert len(explanation.saliency_map) == len([target_class])
            if overlay:
                assert explanation.saliency_map[target_class].ndim == 3
            else:
                assert explanation.saliency_map[target_class].ndim == 2

        if target_explain_group == TargetExplainGroup.ALL:
            explanation_parameters = ExplanationParameters(
                visualization_parameters=visualization_parameters,
                target_explain_group=target_explain_group,
            )
            explanation = explainer(
                self.image,
                explanation_parameters,
                num_masks=5,
                scale_output=scaling,
            )

            assert explanation is not None
            if overlay:
                assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[model_name]
                assert explanation.shape == (354, 500, 3)
            else:
                assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[model_name]
                assert explanation.shape == (224, 224)
                if scaling:
                    for map_ in explanation.saliency_map.values():
                        assert map_.min() == 0, f"{map_.min()}"
                        assert map_.max() in {254, 255}, f"{map_.max()}"
                    if model_name in self._ref_sal_maps:
                        actual_sal_vals = explanation.saliency_map[0][0, :10].astype(np.int16)
                        ref_sal_vals = self._ref_sal_maps[DEFAULT_CLS_MODEL].astype(np.uint8)
                        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_classification_black_box_xai_model_as_input(self):
        retrieve_otx_model(self.data_dir, DEFAULT_CLS_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        model_xai = xai.insert_xai(
            model,
            task=Task.CLASSIFICATION,
        )
        assert has_xai(model_xai), "Updated IR model should has XAI head."

        explainer = Explainer(
            model=model_xai,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=ExplainMode.BLACKBOX,
        )
        explanation_parameters = ExplanationParameters(
            visualization_parameters=VisualizationParameters(overlay=False),
            target_explain_labels=[0],
        )
        explanation = explainer(
            self.image,
            explanation_parameters,
            num_masks=5,
        )

        actual_sal_vals = explanation.saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[DEFAULT_CLS_MODEL].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
