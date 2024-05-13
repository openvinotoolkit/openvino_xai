# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import addict
import cv2
import numpy as np
import openvino
import openvino.runtime as ov
import pytest

from openvino_xai.algorithms.white_box.create_method import (
    create_white_box_detection_explain_method,
)
from openvino_xai.algorithms.white_box.white_box_methods import (
    DetClassProbabilityMapXAIMethod,
)
from openvino_xai.common.parameters import TaskType, XAIMethodType
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explanation.explain import Explainer
from openvino_xai.explanation.explanation_parameters import (
    ExplainMode,
    ExplanationParameters,
    PostProcessParameters,
    TargetExplainGroup,
)
from openvino_xai.explanation.utils import get_preprocess_fn
from openvino_xai.insertion.insertion_parameters import DetectionInsertionParameters

MODEL_CONFIGS = addict.Addict(
    {
        "det_mobilenetv2_atss_bccd": {
            "anchors": None,  # [1, 1, 1, 1, 1],
            "num_classes": 3,
            "node_names": [
                "/bbox_head/atss_cls_1/Conv/WithoutBiases",
                "/bbox_head/atss_cls_2/Conv/WithoutBiases",
                "/bbox_head/atss_cls_3/Conv/WithoutBiases",
                "/bbox_head/atss_cls_4/Conv/WithoutBiases",
            ],
            "input_size": (992, 736),
        },
        "det_mobilenetv2_ssd_bccd": {
            "anchors": [4, 5],
            "num_classes": 4,
            "node_names": [
                "/bbox_head/cls_convs.0/cls_convs.0.3/Conv/WithoutBiases",
                "/bbox_head/cls_convs.1/cls_convs.1.3/Conv/WithoutBiases",
            ],
            "input_size": (864, 864),
        },
        "det_yolox_bccd": {
            "anchors": None,  # [1, 1, 1, 1, 1],
            "num_classes": 3,
            "node_names": [
                "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
                "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
                "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
            ],
            "input_size": (416, 416),
        },
    }
)

MODELS = list(MODEL_CONFIGS.keys())

DEFAULT_MODEL = "det_mobilenetv2_atss_bccd"

TARGET_EXPLAIN_GROUPS = [
    TargetExplainGroup.ALL,
    TargetExplainGroup.CUSTOM,
]


class TestDetWB:
    """
    Tests detection models in WB mode.
    """

    image = cv2.imread("tests/assets/blood.jpg")
    data_dir = Path(".data")
    _ref_sal_maps_reciprocam = {
        "det_mobilenetv2_atss_bccd": np.array([222, 243, 232, 229, 221, 217, 237, 246, 252, 255], dtype=np.uint8),
        "det_mobilenetv2_ssd_bccd": np.array([83, 93, 61, 48, 110, 109, 78, 128, 158, 111], dtype=np.uint8),
        "det_yolox_bccd": np.array([17, 13, 15, 60, 94, 52, 61, 47, 8, 40], dtype=np.uint8),
    }
    _sal_map_size = (23, 23)

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    @pytest.mark.parametrize("target_explain_group", TARGET_EXPLAIN_GROUPS)
    def test_detclassprobabilitymap(self, model_name, embed_normalization, target_explain_group):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)

        cls_head_output_node_names = MODEL_CONFIGS[model_name].node_names
        insertion_parameters = DetectionInsertionParameters(
            embed_normalization=embed_normalization,
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[model_name].anchors,
            saliency_map_size=self._sal_map_size,
            explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
        )

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[model_name].input_size,
            hwc_to_chw=True,
        )
        explainer = Explainer(
            model=model,
            task_type=TaskType.DETECTION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
            insertion_parameters=insertion_parameters,
        )

        target_class_list = [1] if target_explain_group == TargetExplainGroup.CUSTOM else None
        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            target_explain_labels=target_class_list,
            # w/o postrocessing
            post_processing_parameters=PostProcessParameters(),
        )
        explanation = explainer(self.image, explanation_parameters)
        assert explanation is not None

        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanation.saliency_map) == MODEL_CONFIGS[model_name].num_classes
            assert explanation.saliency_map[0].shape == self._sal_map_size

            actual_sal_vals = explanation.saliency_map[0][0, :10].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps_reciprocam[model_name].astype(np.uint8)
            if embed_normalization:
                # Reference values generated with embed_normalization=True
                assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
            else:
                assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100

        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = target_class_list[0]
            assert target_class in explanation.saliency_map
            assert len(explanation.saliency_map) == len(target_class_list)
            assert explanation.saliency_map[target_class].ndim == 2
            assert explanation.saliency_map[target_class].shape == self._sal_map_size

    @pytest.mark.parametrize("target_explain_group", TARGET_EXPLAIN_GROUPS)
    def test_detection_postprocessing(self, target_explain_group):
        model, insertion_parameters = self.get_default_model_and_insertion_parameters()

        target_class_list = [1] if target_explain_group == TargetExplainGroup.CUSTOM else None
        post_processing_parameters = PostProcessParameters(overlay=True)

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[DEFAULT_MODEL].input_size,
            hwc_to_chw=True,
        )
        explainer = Explainer(
            model=model,
            task_type=TaskType.DETECTION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
            insertion_parameters=insertion_parameters,
        )

        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            target_explain_labels=target_class_list,
            post_processing_parameters=post_processing_parameters,
        )
        explanation = explainer(self.image, explanation_parameters)
        assert explanation is not None
        assert explanation.sal_map_shape == (480, 640, 3)
        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanation.saliency_map) == MODEL_CONFIGS[DEFAULT_MODEL].num_classes
        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = target_class_list[0]
            assert len(explanation.saliency_map) == len(target_class_list)
            assert target_class in explanation.saliency_map

    def test_two_sequential_norms(self):
        model, insertion_parameters = self.get_default_model_and_insertion_parameters()

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[DEFAULT_MODEL].input_size,
            hwc_to_chw=True,
        )
        explainer = Explainer(
            model=model,
            task_type=TaskType.DETECTION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
            insertion_parameters=insertion_parameters,
        )

        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.ALL,
            post_processing_parameters=PostProcessParameters(normalize=True),
        )
        explanation = explainer(self.image, explanation_parameters)

        actual_sal_vals = explanation.saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps_reciprocam[DEFAULT_MODEL].astype(np.uint8)
        # Reference values generated with embed_normalization=True
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        for map_ in explanation.saliency_map.values():
            assert map_.min() == 0, f"{map_.min()}"
            assert map_.max() in {254, 255}, f"{map_.max()}"

    def test_create_white_box_detection_explain_method(self):
        """Test create_white_box_detection_explain_method."""
        model, insertion_parameters = self.get_default_model_and_insertion_parameters()

        det_xai_method = create_white_box_detection_explain_method(model, insertion_parameters)
        assert isinstance(det_xai_method, DetClassProbabilityMapXAIMethod)
        assert isinstance(det_xai_method.model_ori, openvino.runtime.Model)

    def get_default_model_and_insertion_parameters(self):
        retrieve_otx_model(self.data_dir, DEFAULT_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_MODEL + ".xml")
        model = ov.Core().read_model(model_path)

        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_MODEL].node_names
        insertion_parameters = DetectionInsertionParameters(
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
            explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
        )
        return model, insertion_parameters
