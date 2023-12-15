# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import pytest

import openvino.runtime as ov

import openvino_xai as ovxai
from openvino_xai.explanation.model_inferrer import ClassificationModelInferrer, ActivationType
from openvino_xai.common.utils import retrieve_otx_model, has_xai
from openvino_xai.explanation.explanation_parameters import PostProcessParameters, TargetExplainGroup, \
    ExplanationParameters, ExplainMode
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters
from openvino_xai.common.parameters import XAIMethodType, TaskType


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


DEFAULT_MODEL = "mlc_mobilenetv3_large_voc"


class TestClsWB:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    data_dir = Path(".data")
    _ref_sal_maps_reciprocam = {
        "mlc_mobilenetv3_large_voc": np.array([215, 214, 233, 239, 218, 206, 210], dtype=np.uint8),
        "mlc_efficient_b0_voc": np.array([73, 242, 156, 219, 197, 239, 69], dtype=np.uint8),
        "mlc_efficient_v2s_voc": np.array([157, 166, 166, 181, 162, 151, 147], dtype=np.uint8),
        "classification_model_with_xai_head": np.array([172, 173, 235, 236, 237, 238, 227], dtype=np.uint8),
    }
    _ref_sal_maps_vitreciprocam = {
        "deit-tiny": np.array([212, 208, 221, 236, 230, 234, 237, 242, 221, 199, 179, 148, 140, 166], dtype=np.uint8)
    }
    _ref_sal_maps_activationmap = {
        "mlc_mobilenetv3_large_voc": np.array([7, 7, 13, 16, 3, 0, 5], dtype=np.uint8),
    }

    @pytest.mark.parametrize("embed_normalization", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.PREDICTIONS,
            TargetExplainGroup.CUSTOM,
        ],
    )
    def test_vitreciprocam(self, embed_normalization, target_explain_group):
        model_name = "deit-tiny"
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")

        model = ov.Core().read_model(model_path)
        insertion_parameters = ClassificationInsertionParameters(
            embed_normalization=embed_normalization,
            explain_method_type=XAIMethodType.VITRECIPROCAM,
        )
        model_xai = ovxai.insert_xai(
            model,
            task_type=TaskType.CLASSIFICATION,
            insertion_parameters=insertion_parameters,
        )
        model_inferrer = ClassificationModelInferrer(model_xai, change_channel_order=True, activation=ActivationType.NONE)

        if target_explain_group == TargetExplainGroup.ALL:
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                post_processing_parameters=PostProcessParameters(),
            )
            explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters)
            assert explanations is not None
            assert len(explanations.saliency_map) == MODEL_NUM_CLASSES[model_name]
            if model_name in self._ref_sal_maps_vitreciprocam:
                actual_sal_vals = explanations.saliency_map[0][0, :].astype(np.int16)
                ref_sal_vals = self._ref_sal_maps_vitreciprocam[model_name].astype(np.uint8)
                if embed_normalization:
                    # Reference values generated with embed_normalization=True
                    assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
                else:
                    assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100
        if target_explain_group == TargetExplainGroup.PREDICTIONS:
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                post_processing_parameters=PostProcessParameters(),
            )
            explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters)
            assert explanations is not None
            assert len(explanations.saliency_map) == len(explanations.prediction)
        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = 1
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                custom_target_indices=[target_class],
                post_processing_parameters=PostProcessParameters(),
            )
            explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters)
            assert explanations is not None
            assert target_class in explanations.saliency_map
            assert len(explanations.saliency_map) == len([target_class])
            assert explanations.saliency_map[target_class].ndim == 2

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.PREDICTIONS,
            TargetExplainGroup.CUSTOM,
        ],
    )
    def test_reciprocam(self, model_name, embed_normalization, target_explain_group):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)
        insertion_parameters = ClassificationInsertionParameters(
            embed_normalization=embed_normalization,
            explain_method_type=XAIMethodType.RECIPROCAM,
        )
        model_xai = ovxai.insert_xai(
            model,
            task_type=TaskType.CLASSIFICATION,
            insertion_parameters=insertion_parameters,
        )
        model_inferrer = ClassificationModelInferrer(model_xai, change_channel_order=True, activation=ActivationType.SIGMOID)

        if target_explain_group == TargetExplainGroup.ALL:
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                post_processing_parameters=PostProcessParameters(),
            )
            explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters)
            assert explanations is not None
            assert len(explanations.saliency_map) == MODEL_NUM_CLASSES[model_name]
            if model_name in self._ref_sal_maps_reciprocam:
                actual_sal_vals = explanations.saliency_map[0][0, :].astype(np.int16)
                ref_sal_vals = self._ref_sal_maps_reciprocam[model_name].astype(np.uint8)
                if embed_normalization:
                    # Reference values generated with embed_normalization=True
                    assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
                else:
                    if model_name == "classification_model_with_xai_head":
                        pytest.skip("model already has fixed xai head - this test cannot change it.")
                    assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100
        if target_explain_group == TargetExplainGroup.PREDICTIONS:
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                post_processing_parameters=PostProcessParameters(),
            )
            explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters)
            assert explanations is not None
            assert len(explanations.saliency_map) == len(explanations.prediction)
        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = 1
            explanation_parameters = ExplanationParameters(
                target_explain_group=target_explain_group,
                custom_target_indices=[target_class],
                post_processing_parameters=PostProcessParameters(),
            )
            explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters)
            assert explanations is not None
            assert target_class in explanations.saliency_map
            assert len(explanations.saliency_map) == len([target_class])
            assert explanations.saliency_map[target_class].ndim == 2


    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    def test_activationmap(self, model_name, embed_normalization):
        if model_name == "classification_model_with_xai_head":
            pytest.skip("model already has reciprocam xai head - this test cannot change it.")
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)
        insertion_parameters = ClassificationInsertionParameters(
            embed_normalization=embed_normalization,
            explain_method_type=XAIMethodType.ACTIVATIONMAP,
        )
        model_xai = ovxai.insert_xai(
            model,
            task_type=TaskType.CLASSIFICATION,
            insertion_parameters=insertion_parameters,
        )
        model_inferrer = ClassificationModelInferrer(model_xai, change_channel_order=True, activation=ActivationType.SIGMOID)

        explanation_parameters = ExplanationParameters(
            post_processing_parameters=PostProcessParameters(),
        )
        explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters=explanation_parameters)
        if model_name in self._ref_sal_maps_activationmap and embed_normalization:
            actual_sal_vals = explanations.saliency_map["per_image_map"][0, :].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps_activationmap[model_name].astype(np.uint8)
            # Reference values generated with embed_normalization=True
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        assert explanations is not None
        assert "per_image_map" in explanations.saliency_map
        assert explanations.saliency_map["per_image_map"].ndim == 2

    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.PREDICTIONS,
            TargetExplainGroup.CUSTOM,
        ],
    )
    @pytest.mark.parametrize("overlay", [True, False])
    def test_classification_postprocessing(self, target_explain_group, overlay):
        retrieve_otx_model(self.data_dir, DEFAULT_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        model_xai = ovxai.insert_xai(
            model,
            task_type=TaskType.CLASSIFICATION,
        )
        model_inferrer = ClassificationModelInferrer(model_xai, change_channel_order=True, activation=ActivationType.SIGMOID)

        explain_targets = None
        if target_explain_group == TargetExplainGroup.CUSTOM:
            explain_targets = [1]
        post_processing_parameters = PostProcessParameters(overlay=overlay)

        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            custom_target_indices=explain_targets,
            post_processing_parameters=post_processing_parameters,
        )
        explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters=explanation_parameters)
        assert explanations is not None
        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanations.saliency_map) == MODEL_NUM_CLASSES[DEFAULT_MODEL]
        if target_explain_group == TargetExplainGroup.PREDICTIONS:
            assert len(explanations.saliency_map) == len(explanations.prediction)
        if target_explain_group == TargetExplainGroup.CUSTOM:
            assert len(explanations.saliency_map) == len(explain_targets)
            assert 1 in explanations.saliency_map
        if overlay:
            assert explanations.sal_map_shape == (354, 500, 3)
        else:
            assert explanations.sal_map_shape == (7, 7)
            for map_ in explanations.saliency_map.values():
                assert map_.min() == 0, f"{map_.min()}"
                assert map_.max() in {254, 255}, f"{map_.max()}"

    def test_two_sequential_norms(self):
        retrieve_otx_model(self.data_dir, DEFAULT_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        model_xai = ovxai.insert_xai(
            model,
            task_type=TaskType.CLASSIFICATION,
        )
        model_inferrer = ClassificationModelInferrer(model_xai, change_channel_order=True, activation=ActivationType.SIGMOID)

        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.ALL,
            post_processing_parameters=PostProcessParameters(normalize=True),
        )
        explanations = ovxai.explain(model_inferrer, self.image, explanation_parameters=explanation_parameters)

        actual_sal_vals = explanations.saliency_map[0][0, :].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps_reciprocam[DEFAULT_MODEL].astype(np.uint8)
        # Reference values generated with embed_normalization=True
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        for map_ in explanations.saliency_map.values():
            assert map_.min() == 0, f"{map_.min()}"
            assert map_.max() in {254, 255}, f"{map_.max()}"


class TestClsBB:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    data_dir = Path(".data")
    _ref_sal_maps = {
        "mlc_mobilenetv3_large_voc": np.array([13, 18, 23, 29, 34, 40, 45, 51, 57, 65], dtype=np.uint8),
        "mlc_efficient_b0_voc": np.array([9, 14, 20, 25, 31, 37, 43, 48, 55, 63], dtype=np.uint8),
        "mlc_efficient_v2s_voc": np.array([14, 19, 24, 30, 36, 41, 47, 53, 60, 67], dtype=np.uint8),
        "classification_model_with_xai_head": np.array([8, 13, 18, 24, 29, 35, 40, 46, 52, 60], dtype=np.uint8),
    }

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("overlay", [True, False])
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.CUSTOM,
        ],
    )
    @pytest.mark.parametrize("normalize", [True, False])
    def test_classification_black_box_postprocessing(self, model_name, overlay, target_explain_group, normalize):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)
        model_inferrer = ClassificationModelInferrer(model, change_channel_order=True, activation=ActivationType.SIGMOID)

        post_processing_parameters = PostProcessParameters(
            overlay=overlay,
        )

        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = 1
            explanation_parameters = ExplanationParameters(
                explain_mode=ExplainMode.BLACKBOX,
                post_processing_parameters=post_processing_parameters,
                target_explain_group=target_explain_group,
                custom_target_indices=[target_class],
                black_box_method_kwargs={"num_masks": 5, "asynchronous_inference": False, "normalize": normalize}
            )
            explanation = ovxai.explain(
                model_inferrer,
                self.image,
                explanation_parameters=explanation_parameters,
            )

            assert explanation is not None
            assert target_class in explanation.saliency_map
            assert len(explanation.saliency_map) == len([target_class])
            if overlay:
                assert explanation.saliency_map[target_class].ndim == 3
            else:
                assert explanation.saliency_map[target_class].ndim == 2
        else:
            explanation_parameters = ExplanationParameters(
                explain_mode=ExplainMode.BLACKBOX,
                post_processing_parameters=post_processing_parameters,
                target_explain_group=target_explain_group,
                black_box_method_kwargs={"num_masks": 5, "asynchronous_inference": False, "normalize": normalize}
            )
            explanation = ovxai.explain(
                model_inferrer,
                self.image,
                explanation_parameters=explanation_parameters,
            )

            assert explanation is not None
            if overlay:
                assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[model_name]
                assert explanation.sal_map_shape == (354, 500, 3)
            else:
                assert len(explanation.saliency_map) == MODEL_NUM_CLASSES[model_name]
                assert explanation.sal_map_shape == (224, 224)
                if normalize:
                    for map_ in explanation.saliency_map.values():
                        assert map_.min() == 0, f"{map_.min()}"
                        assert map_.max() in {254, 255}, f"{map_.max()}"

    @pytest.mark.parametrize("model_name", MODELS_VOC)
    @pytest.mark.parametrize("asynchronous_inference", [True, False])
    @pytest.mark.parametrize("throughput_inference", [True, False])
    def test_classification_black_box_pred_class(self, model_name, asynchronous_inference, throughput_inference):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)
        model_inferrer = ClassificationModelInferrer(model, change_channel_order=True, activation=ActivationType.SIGMOID)

        explanation_parameters = ExplanationParameters(
            explain_mode=ExplainMode.BLACKBOX,
            post_processing_parameters=PostProcessParameters(overlay=False),
            black_box_method_kwargs={
                "num_masks": 5, "asynchronous_inference": False, "throughput_inference": False
            }
        )

        explanation = ovxai.explain(
            model_inferrer,
            self.image,
            explanation_parameters=explanation_parameters,
        )
        assert explanation is not None
        assert len(explanation.saliency_map) > 0
        assert len(explanation.saliency_map) == len(explanation.prediction)
        assert explanation.sal_map_shape == (224, 224)

        # Check that returned saliency map classes and predicted classes are the same
        predicted_class_idx = sorted([pred[0] for pred in explanation.prediction])
        returned_sal_map_classes = list(sorted(explanation.saliency_map.keys()))
        assert predicted_class_idx == returned_sal_map_classes

        if model_name in self._ref_sal_maps:
            first_idx = predicted_class_idx[0]
            actual_sal_vals = explanation.saliency_map[first_idx][0, :10].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps[model_name].astype(np.uint8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_classification_black_box_xai_model_as_input(self):
        retrieve_otx_model(self.data_dir, DEFAULT_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        model_xai = ovxai.insert_xai(
            model,
            task_type=TaskType.CLASSIFICATION,
        )
        assert has_xai(model_xai), "Updated IR model should has XAI head."
        model_inferrer = ClassificationModelInferrer(model_xai, change_channel_order=True, activation=ActivationType.SIGMOID)

        explanation_parameters = ExplanationParameters(
            explain_mode=ExplainMode.BLACKBOX,
            post_processing_parameters=PostProcessParameters(overlay=False),
            black_box_method_kwargs={"num_masks": 5}
        )
        explanation = ovxai.explain(
            model_inferrer,
            self.image,
            explanation_parameters=explanation_parameters,
        )

        predicted_class_idx = sorted([pred[0] for pred in explanation.prediction])
        first_idx = predicted_class_idx[0]
        actual_sal_vals = explanation.saliency_map[first_idx][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[DEFAULT_MODEL].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
