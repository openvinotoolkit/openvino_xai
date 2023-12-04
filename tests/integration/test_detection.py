from pathlib import Path

import addict
import cv2
import numpy as np
import pytest

import openvino
import openvino.model_api as mapi

import openvino_xai as ovxai
from openvino_xai.algorithms.white_box.create_method import create_white_box_detection_explain_method
from openvino_xai.algorithms.white_box.white_box_methods import DetClassProbabilityMapXAIMethod
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.common.parameters import XAIMethodType
from openvino_xai.explanation.explanation_parameters import (
    PostProcessParameters,
    TargetExplainGroup,
    ExplanationParameters,
)
from openvino_xai.insertion.insertion_parameters import DetectionInsertionParameters

MODEL_CONFIGS = addict.Addict(
    {
        "det_mobilenetv2_atss_bccd": 
            {
                "anchors": [1, 1, 1, 1, 1],
                "num_classes": 3,
                "node_names": [
                    "/bbox_head/atss_cls_1/Conv/WithoutBiases",
                    "/bbox_head/atss_cls_2/Conv/WithoutBiases",
                    "/bbox_head/atss_cls_3/Conv/WithoutBiases",
                    "/bbox_head/atss_cls_4/Conv/WithoutBiases",
                ],
            },
        "det_mobilenetv2_ssd_bccd": 
            {
                "anchors": [4, 5],
                "num_classes": 4,
                "node_names": [
                    "/bbox_head/cls_convs.0/cls_convs.0.3/Conv/WithoutBiases",
                    "/bbox_head/cls_convs.1/cls_convs.1.3/Conv/WithoutBiases",
                ],
            },
        "det_yolox_bccd": 
            {
                "anchors": [1, 1, 1, 1, 1],
                "num_classes": 3,
                "node_names": [
                    "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
                    "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
                    "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
                ],
            },     
    }
)

MODELS = list(MODEL_CONFIGS.keys())

DEFAULT_MODEL = "det_mobilenetv2_atss_bccd"

TARGET_EXPLAIN_GROUPS = [
    TargetExplainGroup.ALL,
    TargetExplainGroup.PREDICTIONS,
    TargetExplainGroup.CUSTOM,
]


def get_mapi_model_wrapper(model_name: str, data_dir: Path = Path(".data")) -> mapi.models.Model:
    """Download model and create model wrapper for it.

    Args:
        model_name (str): Name of model
        data_dir (int): Dir where to download model

    Returns:
        model_wrapper (mapi.models.Model)
    """
    assert model_name in MODELS, f"Test model should be in list of supported models {MODELS}"
    retrieve_otx_model(data_dir, model_name)
    model_path = data_dir / "otx_models" / (model_name + ".xml")

    model_wrapper = mapi.models.DetectionModel.create_model(
        model_path,
        model_type="ssd",
    )
    return model_wrapper


class TestDetWB:
    """
    Tests detection models in WB mode.
    """

    image = cv2.imread("tests/assets/blood_image.jpg")
    data_dir = Path(".data")
    _ref_sal_maps_reciprocam = {
        "det_mobilenetv2_atss_bccd": np.array([226, 247, 235, 231, 222, 217, 236, 246, 251, 255], dtype=np.uint8),
        "det_mobilenetv2_ssd_bccd": np.array([84, 95, 61, 47, 111, 111, 77, 127, 153, 108], dtype=np.uint8),
        "det_yolox_bccd": np.array([17, 12, 15, 63, 100, 55, 63, 47, 8, 40], dtype=np.uint8),
    }
    _sal_map_size = (23, 23)

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_normalization", [True, False])
    @pytest.mark.parametrize("target_explain_group", TARGET_EXPLAIN_GROUPS)
    def test_detclassprobabilitymap(self, model_name, embed_normalization, target_explain_group):
        model_wrapper = get_mapi_model_wrapper(model_name)
        cls_head_output_node_names = MODEL_CONFIGS[model_name].node_names
        insertion_parameters = DetectionInsertionParameters(
            embed_normalization=embed_normalization,
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[model_name].anchors,
            saliency_map_size=self._sal_map_size,
            explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
        )
        mapi_wrapper_xai = ovxai.insertion.insert_xai_into_mapi_wrapper(
            model_wrapper, insertion_parameters=insertion_parameters
        )
        target_class_list = [1] if target_explain_group == TargetExplainGroup.CUSTOM else None

        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            custom_target_indices=target_class_list,
            # w/o postrocessing
            post_processing_parameters=PostProcessParameters(),
        )
        explanations = ovxai.explain(mapi_wrapper_xai, self.image, explanation_parameters)

        # For SSD add a background class
        background_class = model_name == "det_mobilenetv2_ssd_bccd"
        assert explanations is not None
        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanations.saliency_map) == len(mapi_wrapper_xai.labels) + background_class
            assert explanations.saliency_map[0].shape == self._sal_map_size

            actual_sal_vals = explanations.saliency_map[0][0, :10].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps_reciprocam[model_name].astype(np.uint8)
            if embed_normalization:
                # Reference values generated with embed_normalization=True
                assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
            else:
                assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100

        if target_explain_group == TargetExplainGroup.PREDICTIONS:
            assert len(explanations.saliency_map) == len(explanations.prediction)

        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = target_class_list[0]
            assert target_class in explanations.saliency_map
            assert len(explanations.saliency_map) == len(target_class_list)
            assert explanations.saliency_map[target_class].ndim == 2
            assert explanations.saliency_map[target_class].shape == self._sal_map_size

    @pytest.mark.parametrize("target_explain_group", TARGET_EXPLAIN_GROUPS)
    def test_detection_postprocessing(self, target_explain_group):
        model_wrapper = get_mapi_model_wrapper(DEFAULT_MODEL)
        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_MODEL].node_names
        insertion_parameters = DetectionInsertionParameters(
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
            explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
        )
        mapi_wrapper_xai = ovxai.insertion.insert_xai_into_mapi_wrapper(
            model_wrapper, insertion_parameters=insertion_parameters
        )

        target_class_list = [1] if target_explain_group == TargetExplainGroup.CUSTOM else None
        post_processing_parameters = PostProcessParameters(overlay=True)

        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            custom_target_indices=target_class_list,
            post_processing_parameters=post_processing_parameters,
        )
        explanations = ovxai.explain(mapi_wrapper_xai, self.image, explanation_parameters=explanation_parameters)
        assert explanations is not None
        assert explanations.sal_map_shape == (480, 640, 3)
        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanations.saliency_map) == MODEL_CONFIGS[DEFAULT_MODEL].num_classes
        if target_explain_group == TargetExplainGroup.PREDICTIONS:
            assert len(explanations.saliency_map) == len(explanations.prediction)
        if target_explain_group == TargetExplainGroup.CUSTOM:
            target_class = target_class_list[0]
            assert len(explanations.saliency_map) == len(target_class_list)
            assert target_class in explanations.saliency_map

    def test_two_sequential_norms(self):
        model_wrapper = get_mapi_model_wrapper(DEFAULT_MODEL)
        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_MODEL].node_names
        insertion_parameters = DetectionInsertionParameters(
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
            explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
        )
        mapi_wrapper_xai = ovxai.insertion.insert_xai_into_mapi_wrapper(
            model_wrapper, insertion_parameters=insertion_parameters
        )

        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.ALL,
            post_processing_parameters=PostProcessParameters(normalize=True),
        )
        explanations = ovxai.explain(mapi_wrapper_xai, self.image, explanation_parameters=explanation_parameters)

        actual_sal_vals = explanations.saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps_reciprocam[DEFAULT_MODEL].astype(np.uint8)
        # Reference values generated with embed_normalization=True
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        for map_ in explanations.saliency_map.values():
            assert map_.min() == 0, f"{map_.min()}"
            assert map_.max() in {254, 255}, f"{map_.max()}"

    def test_create_white_box_detection_explain_method(self):
        """Test create_white_box_detection_explain_method."""
        model_wrapper = get_mapi_model_wrapper(DEFAULT_MODEL)
        model = model_wrapper.get_model()

        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_MODEL].node_names
        insertion_parameters = DetectionInsertionParameters(
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
            explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
        )

        detXAImethod = create_white_box_detection_explain_method(model, insertion_parameters)
        assert isinstance(detXAImethod, DetClassProbabilityMapXAIMethod)
        assert isinstance(detXAImethod.model_ori, openvino.runtime.Model)
