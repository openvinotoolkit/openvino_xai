# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404 (not a part of product)
from pathlib import Path

import addict
import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import get_preprocess_fn
from openvino_xai.methods.black_box.aise.detection import AISEDetection
from openvino_xai.methods.factory import BlackBoxMethodFactory, WhiteBoxMethodFactory
from openvino_xai.methods.white_box.det_class_probability_map import (
    DetClassProbabilityMap,
)

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

DEFAULT_DET_MODEL = "det_mobilenetv2_atss_bccd"
FAST_DET_MODEL = "det_mobilenetv2_atss_bccd"

EXPLAIN_ALL_CLASSES = [
    True,
    False,
]


class TestDetWB:
    """
    Tests detection models in white-box mode.
    """

    image = cv2.imread("tests/assets/blood.jpg")
    _ref_sal_maps = {
        "det_mobilenetv2_atss_bccd": np.array([222, 243, 232, 229, 221, 217, 237, 246, 252, 255], dtype=np.uint8),
        "det_mobilenetv2_ssd_bccd": np.array([83, 93, 61, 48, 110, 109, 78, 128, 158, 111], dtype=np.uint8),
        "det_yolox_bccd": np.array([17, 13, 15, 60, 94, 52, 61, 47, 8, 40], dtype=np.uint8),
    }
    _sal_map_size = (23, 23)

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("embed_scaling", [True, False])
    @pytest.mark.parametrize("explain_all_classes", EXPLAIN_ALL_CLASSES)
    def test_detclassprobabilitymap(self, model_name, embed_scaling, explain_all_classes):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)

        cls_head_output_node_names = MODEL_CONFIGS[model_name].node_names
        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[model_name].input_size,
            hwc_to_chw=True,
        )
        explainer = Explainer(
            model=model,
            task=Task.DETECTION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
            explain_method=Method.DETCLASSPROBABILITYMAP,
            target_layer=cls_head_output_node_names,
            embed_scaling=embed_scaling,
            num_anchors=MODEL_CONFIGS[model_name].anchors,
            saliency_map_size=self._sal_map_size,
        )

        target_class_list = [1] if not explain_all_classes else -1
        explanation = explainer(
            self.image,
            targets=target_class_list,
            resize=False,
            colormap=False,
        )
        assert explanation is not None

        if explain_all_classes:
            assert len(explanation.saliency_map) == MODEL_CONFIGS[model_name].num_classes
            assert explanation.saliency_map[0].shape == self._sal_map_size

            actual_sal_vals = explanation.saliency_map[0][0, :10].astype(np.int16)
            ref_sal_vals = self._ref_sal_maps[model_name].astype(np.uint8)
            if embed_scaling:
                # Reference values generated with embed_scaling=True
                assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
            else:
                assert np.sum(np.abs(actual_sal_vals - ref_sal_vals)) > 100

        if not explain_all_classes:
            target_class = target_class_list[0]
            assert target_class in explanation.saliency_map
            assert len(explanation.saliency_map) == len(target_class_list)
            assert explanation.saliency_map[target_class].ndim == 2
            assert explanation.saliency_map[target_class].shape == self._sal_map_size

    @pytest.mark.parametrize("explain_all_classes", EXPLAIN_ALL_CLASSES)
    def test_detection_visualizing(self, explain_all_classes):
        model = self.get_default_model()

        target_class_list = [1] if not explain_all_classes else -1

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[DEFAULT_DET_MODEL].input_size,
            hwc_to_chw=True,
        )
        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names
        explainer = Explainer(
            model=model,
            task=Task.DETECTION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
            explain_method=Method.DETCLASSPROBABILITYMAP,
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
        )

        explanation = explainer(
            self.image,
            targets=target_class_list,
            overlay=True,
        )
        assert explanation is not None
        assert explanation.shape == (480, 640, 3)
        if explain_all_classes:
            assert len(explanation.saliency_map) == MODEL_CONFIGS[DEFAULT_DET_MODEL].num_classes
        if not explain_all_classes:
            target_class = target_class_list[0]
            assert len(explanation.saliency_map) == len(target_class_list)
            assert target_class in explanation.saliency_map

    def test_two_sequential_norms(self):
        model = self.get_default_model()

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[DEFAULT_DET_MODEL].input_size,
            hwc_to_chw=True,
        )
        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names
        explainer = Explainer(
            model=model,
            task=Task.DETECTION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
            explain_method=Method.DETCLASSPROBABILITYMAP,
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
        )

        explanation = explainer(
            self.image,
            targets=-1,
            scaling=True,
            resize=False,
            colormap=False,
        )

        actual_sal_vals = explanation.saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = self._ref_sal_maps[DEFAULT_DET_MODEL].astype(np.uint8)
        # Reference values generated with embed_scaling=True
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

        for map_ in explanation.saliency_map.values():
            assert map_.min() == 0, f"{map_.min()}"
            assert map_.max() in {254, 255}, f"{map_.max()}"

    def test_create_white_box_detection_method(self):
        """Test create_white_box_detection_method."""
        model = self.get_default_model()

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[DEFAULT_DET_MODEL].input_size,
            hwc_to_chw=True,
        )
        cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names
        det_xai_method = WhiteBoxMethodFactory.create_method(
            Task.DETECTION,
            model,
            preprocess_fn,
            explain_method=Method.DETCLASSPROBABILITYMAP,
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
            saliency_map_size=self._sal_map_size,
        )
        assert isinstance(det_xai_method, DetClassProbabilityMap)
        assert isinstance(det_xai_method.model_ori, ov.Model)

    def get_default_model(self):
        retrieve_otx_model(self.data_dir, DEFAULT_DET_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        return model


class TestDetBB:
    """
    Tests detection models in black-box mode.
    """

    image = cv2.imread("tests/assets/blood.jpg")

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    @pytest.mark.parametrize("model_name", MODELS)
    def test_aisedetection(self, model_name):
        retrieve_otx_model(self.data_dir, model_name)
        model_path = self.data_dir / "otx_models" / (model_name + ".xml")
        model = ov.Core().read_model(model_path)

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[model_name].input_size,
            hwc_to_chw=True,
        )
        explainer = Explainer(
            model=model,
            task=Task.DETECTION,
            preprocess_fn=preprocess_fn,
            postprocess_fn=self.postprocess_fn,
            explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
            num_iterations_per_kernel=5,
            divisors=[5],
        )

        target_list = [1]
        explanation = explainer(
            self.image,
            targets=target_list,
            resize=False,
            colormap=False,
        )
        assert explanation is not None

        target_class = target_list[0]
        assert target_class in explanation.saliency_map
        assert len(explanation.saliency_map) == len(target_list)
        assert explanation.saliency_map[target_class].ndim == 2

    def test_detection_visualizing(self):
        model = self.get_default_model()

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[FAST_DET_MODEL].input_size,
            hwc_to_chw=True,
        )
        explainer = Explainer(
            model=model,
            task=Task.DETECTION,
            preprocess_fn=preprocess_fn,
            postprocess_fn=self.postprocess_fn,
            explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
            num_iterations_per_kernel=5,
            divisors=[5],
        )

        target_list = [1]
        explanation = explainer(
            self.image,
            targets=target_list,
            overlay=True,
        )
        assert explanation is not None
        assert explanation.shape == (480, 640, 3)

        target_class = target_list[0]
        assert len(explanation.saliency_map) == len(target_list)
        assert target_class in explanation.saliency_map

    def test_create_aise_detection_method(self):
        """Test create_white_box_detection_method."""
        model = self.get_default_model()

        preprocess_fn = get_preprocess_fn(
            input_size=MODEL_CONFIGS[FAST_DET_MODEL].input_size,
            hwc_to_chw=True,
        )
        det_xai_method = BlackBoxMethodFactory.create_method(
            Task.DETECTION,
            model,
            preprocess_fn,
        )
        assert isinstance(det_xai_method, AISEDetection)

    def get_default_model(self):
        retrieve_otx_model(self.data_dir, FAST_DET_MODEL)
        model_path = self.data_dir / "otx_models" / (FAST_DET_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        return model

    @staticmethod
    def postprocess_fn(x) -> np.ndarray:
        """Returns boxes, scores, labels."""
        return x["boxes"][:, :, :4], x["boxes"][:, :, 4], x["labels"]


class TestExample:
    """Test sanity of examples/run_detection.py."""

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    def test_default_model(self):
        retrieve_otx_model(self.data_dir, DEFAULT_DET_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
        cmd = [
            "python",
            "examples/run_detection.py",
            model_path,
            "tests/assets/blood.jpg",
        ]
        subprocess.run(cmd, check=True)  # noqa: S603, PLW1510
