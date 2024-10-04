# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.base import Prediction
from openvino_xai.methods.black_box.aise.classification import AISEClassification
from openvino_xai.methods.black_box.aise.detection import AISEDetection
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.methods.black_box.rise import RISE
from openvino_xai.methods.black_box.utils import (
    check_classification_output,
    check_detection_output,
)
from tests.intg.test_classification import DEFAULT_CLS_MODEL
from tests.intg.test_detection import DEFAULT_DET_MODEL


class InputSampling:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn()

    def get_cls_model(self, fxt_data_root):
        retrieve_otx_model(fxt_data_root, DEFAULT_CLS_MODEL)
        model_path = fxt_data_root / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        return ov.Core().read_model(model_path)

    def get_det_model(self, fxt_data_root):
        detection_model = "det_yolox_bccd"
        retrieve_otx_model(fxt_data_root, detection_model)
        model_path = fxt_data_root / "otx_models" / (detection_model + ".xml")
        return ov.Core().read_model(model_path)

    def _generate_with_preset(self, method, preset):
        _ = method.generate_saliency_map(
            data=self.image,
            target_indices=[1],
            preset=preset,
        )

    @staticmethod
    def preprocess_det_fn(x: np.ndarray) -> np.ndarray:
        x = cv2.resize(src=x, dsize=(416, 416))  # OTX YOLOX
        x = x.transpose((2, 0, 1))
        x = np.expand_dims(x, 0)
        return x

    @staticmethod
    def postprocess_det_fn(x) -> np.ndarray:
        """Returns boxes, scores, labels."""
        return x["boxes"][:, :, :4], x["boxes"][:, :, 4], x["labels"]


class TestAISEClassification(InputSampling):
    @pytest.mark.parametrize("target_indices", [[0], [0, 1]])
    def test_run(self, target_indices, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)

        aise_method = AISEClassification(model, self.postprocess_fn, self.preprocess_fn)
        saliency_map = aise_method.generate_saliency_map(
            data=self.image,
            target_indices=target_indices,
            preset=Preset.SPEED,
            num_iterations_per_kernel=10,
            kernel_widths=[0.1],
        )
        assert aise_method.num_iterations_per_kernel == 10
        assert aise_method.kernel_widths == [0.1]

        assert isinstance(saliency_map, dict)
        assert len(saliency_map) == len(target_indices)
        for target in target_indices:
            assert target in saliency_map
            assert target in aise_method.predictions
            assert isinstance(aise_method.predictions[target], Prediction)

        ref_target = 0
        assert saliency_map[ref_target].dtype == np.uint8
        assert saliency_map[ref_target].shape == (224, 224)
        assert (saliency_map[ref_target] >= 0).all() and (saliency_map[ref_target] <= 255).all()

        actual_sal_vals = saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = np.array([68, 69, 69, 69, 70, 70, 71, 71, 72, 72], dtype=np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_preset(self, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)
        method = AISEClassification(model, self.postprocess_fn, self.preprocess_fn)

        tic = time.time()
        self._generate_with_preset(method, Preset.SPEED)
        toc = time.time()
        time_speed = toc - tic
        assert method.num_iterations_per_kernel == 20
        assert np.all(method.kernel_widths == np.array([0.1, 0.175, 0.25]))

        tic = time.time()
        self._generate_with_preset(method, Preset.BALANCE)
        toc = time.time()
        time_balance = toc - tic
        assert method.num_iterations_per_kernel == 50
        assert np.all(method.kernel_widths == np.array([0.1, 0.175, 0.25]))

        tic = time.time()
        self._generate_with_preset(method, Preset.QUALITY)
        toc = time.time()
        time_quality = toc - tic
        assert method.num_iterations_per_kernel == 50
        np.testing.assert_allclose(method.kernel_widths, np.array([0.075, 0.11875, 0.1625, 0.20625, 0.25]))

        assert time_speed < time_balance < time_quality


class TestAISEDetection(InputSampling):
    @pytest.mark.parametrize("target_indices", [[0], [0, 1]])
    def test_run(self, target_indices, fxt_data_root: Path):
        model = self.get_det_model(fxt_data_root)

        aise_method = AISEDetection(model, self.postprocess_det_fn, self.preprocess_det_fn)
        saliency_map = aise_method.generate_saliency_map(
            data=self.image,
            target_indices=target_indices,
            preset=Preset.SPEED,
            num_iterations_per_kernel=10,
            divisors=[5],
        )
        assert aise_method.num_iterations_per_kernel == 10
        assert aise_method.divisors == [5]

        assert isinstance(saliency_map, dict)
        assert len(saliency_map) == len(target_indices)
        for target in target_indices:
            assert target in saliency_map
            assert target in aise_method.predictions
            assert isinstance(aise_method.predictions[target], Prediction)

        ref_target = 0
        assert saliency_map[ref_target].dtype == np.uint8
        assert saliency_map[ref_target].shape == (416, 416)
        assert (saliency_map[ref_target] >= 0).all() and (saliency_map[ref_target] <= 255).all()

        actual_sal_vals = saliency_map[0][150, 240:250].astype(np.int16)
        ref_sal_vals = np.array([152, 168, 184, 199, 213, 225, 235, 243, 247, 249], dtype=np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_target_none(self, fxt_data_root: Path):
        model = self.get_det_model(fxt_data_root)

        aise_method = AISEDetection(model, self.postprocess_det_fn, self.preprocess_det_fn)
        saliency_map = aise_method.generate_saliency_map(
            data=self.image,
            target_indices=None,
            preset=Preset.SPEED,
            num_iterations_per_kernel=1,
            divisors=[5],
        )
        assert len(saliency_map) == 56

    def test_preset(self, fxt_data_root: Path):
        model = self.get_det_model(fxt_data_root)
        method = AISEDetection(model, self.postprocess_det_fn, self.preprocess_det_fn)

        tic = time.time()
        self._generate_with_preset(method, Preset.SPEED)
        toc = time.time()
        time_speed = toc - tic
        assert method.num_iterations_per_kernel == 20
        assert np.all(method.divisors == np.array([7.0, 4.0, 1.0]))

        tic = time.time()
        self._generate_with_preset(method, Preset.BALANCE)
        toc = time.time()
        time_balance = toc - tic
        assert method.num_iterations_per_kernel == 50
        assert np.all(method.divisors == np.array([7.0, 4.0, 1.0]))

        tic = time.time()
        self._generate_with_preset(method, Preset.QUALITY)
        toc = time.time()
        time_quality = toc - tic
        assert method.num_iterations_per_kernel == 50
        assert np.all(method.divisors == np.array([8.0, 6.25, 4.5, 2.75, 1.0]))

        assert time_speed < time_balance < time_quality


class TestRISE(InputSampling):
    @pytest.mark.parametrize("target_indices", [[0], None])
    def test_run(self, target_indices, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)

        rise_method = RISE(model, self.postprocess_fn, self.preprocess_fn)
        saliency_map = rise_method.generate_saliency_map(
            self.image,
            target_indices,
            num_masks=5,
        )

        if target_indices == [0]:
            assert isinstance(saliency_map, dict)
            assert saliency_map[0].dtype == np.uint8
            assert saliency_map[0].shape == (224, 224)
            assert (saliency_map[0] >= 0).all() and (saliency_map[0] <= 255).all()

            actual_sal_vals = saliency_map[0][0, :10].astype(np.int16)
            ref_sal_vals = np.array([246, 241, 236, 231, 226, 221, 216, 211, 205, 197], dtype=np.uint8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

            assert target_indices[0] in rise_method.predictions
            assert isinstance(rise_method.predictions[target_indices[0]], Prediction)
        else:
            isinstance(saliency_map, np.ndarray)
            assert saliency_map.dtype == np.uint8
            assert saliency_map.shape == (1, 20, 224, 224)
            assert (saliency_map >= 0).all() and (saliency_map <= 255).all()

            actual_sal_vals = saliency_map[0][0][0, :10].astype(np.int16)
            ref_sal_vals = np.array([246, 241, 236, 231, 226, 221, 216, 211, 205, 197], dtype=np.uint8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_preset(self, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)
        method = RISE(model, self.postprocess_fn, self.preprocess_fn)

        tic = time.time()
        self._generate_with_preset(method, Preset.SPEED)
        toc = time.time()
        time_speed = toc - tic
        assert method.num_masks == 1000
        assert method.num_cells == 4

        tic = time.time()
        self._generate_with_preset(method, Preset.BALANCE)
        toc = time.time()
        time_balance = toc - tic
        assert method.num_masks == 5000
        assert method.num_cells == 8

        tic = time.time()
        self._generate_with_preset(method, Preset.QUALITY)
        toc = time.time()
        time_quality = toc - tic
        assert method.num_masks == 10_000
        assert method.num_cells == 12

        assert time_speed < time_balance < time_quality


def test_check_classification_output():
    with pytest.raises(Exception) as exc_info:
        x = 1
        check_classification_output(x)
    assert str(exc_info.value) == "Postprocess function should return numpy array."

    with pytest.raises(Exception) as exc_info:
        x = np.zeros((2, 2, 2))
        check_classification_output(x)
    assert str(exc_info.value) == "Postprocess function should return two dimentional numpy array with batch size of 1."


def test_check_detection_output():
    with pytest.raises(Exception) as exc_info:
        x = 1
        check_detection_output(x)
    assert str(exc_info.value) == "Postprocess function should return sized object."

    with pytest.raises(Exception) as exc_info:
        x = 1, 2
        check_detection_output(x)
    assert (
        str(exc_info.value)
        == "Postprocess function should return three containers: boxes (format: [x1, y1, x2, y2]), scores, labels."
    )

    with pytest.raises(Exception) as exc_info:
        x = np.array([1]), np.array([1]), 1
        check_detection_output(x)
    assert str(exc_info.value) == "Postprocess function should return numpy arrays."

    with pytest.raises(Exception) as exc_info:
        x = np.ones((1, 2)), np.ones((1, 2)), np.ones((2, 2))
        check_detection_output(x)
    assert str(exc_info.value) == "Postprocess function should return numpy arrays with batch size of 1."

    with pytest.raises(Exception) as exc_info:
        x = np.ones((1, 2)), np.ones((1)), np.ones((1, 2, 3))
        check_detection_output(x)
    assert str(exc_info.value) == "Boxes should be three-dimentional [Batch, NumBoxes, BoxCoords]."

    with pytest.raises(Exception) as exc_info:
        x = np.ones((1, 2, 4)), np.ones((1)), np.ones((1, 2, 3))
        check_detection_output(x)
    assert str(exc_info.value) == "Scores should be two-dimentional [Batch, Scores]."

    with pytest.raises(Exception) as exc_info:
        x = np.ones((1, 2, 4)), np.ones((1, 2)), np.ones((1, 2, 3))
        check_detection_output(x)
    assert str(exc_info.value) == "Labels should be two-dimentional  [Batch, Labels]."
