# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest
from pytest_mock import MockerFixture

from openvino_xai.api.api import insert_xai
from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.base import Preset
from tests.unit.explainer.test_explanation_utils import VOC_NAMES

MODEL_NAME = "mlc_mobilenetv3_large_voc"


class TestExplainer:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    @pytest.mark.parametrize(
        "explain_mode",
        [
            ExplainMode.WHITEBOX,
            ExplainMode.BLACKBOX,
            ExplainMode.AUTO,
        ],
    )
    @pytest.mark.parametrize("with_xai_originally", [True, False])
    def test_explainer(self, explain_mode, with_xai_originally):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        if with_xai_originally:
            model = insert_xai(
                model,
                task=Task.CLASSIFICATION,
            )

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=explain_mode,
        )

        explanation = explainer(
            self.image,
            targets=[11, 14],
            label_names=VOC_NAMES,  # optional
            preset=Preset.SPEED,
        )

        assert len(explanation.saliency_map) == 2

    @pytest.mark.parametrize(
        "explain_all_classes",
        [
            True,
            False,
        ],
    )
    def test_explainer_all_classes(self, explain_all_classes):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
        )

        if explain_all_classes:
            explanation = explainer(
                self.image,
                targets=-1,
                label_names=VOC_NAMES,  # optional
            )
        else:
            explanation = explainer(
                self.image,
                targets=[11, 14],
                label_names=VOC_NAMES,  # optional
            )

        if explain_all_classes:
            assert len(explanation.saliency_map) == 20
        else:
            assert len(explanation.saliency_map) == 2

    @pytest.mark.parametrize(
        "explain_all_classes",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "explain_method",
        [
            Method.AISE,
            Method.RISE,
        ],
    )
    def test_explainer_all_classes_black_box(self, explain_all_classes, explain_method):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=ExplainMode.BLACKBOX,
            explain_method=explain_method,
        )

        if explain_method == Method.AISE:
            kwargs = {"num_iterations_per_kernel": 2}
        else:
            kwargs = {"num_masks": 2}

        if explain_all_classes:
            explanation = explainer(
                self.image,
                targets=-1,
                **kwargs,
            )
        else:
            explanation = explainer(
                self.image,
                targets=[11, 14],
                **kwargs,
            )

        if explain_all_classes:
            assert len(explanation.saliency_map) == 20
        else:
            assert len(explanation.saliency_map) == 2

    def test_explainer_wo_preprocessing(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")

        # White-box
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model,
            task=Task.CLASSIFICATION,
        )

        processed_data = self.preprocess_fn(self.image)
        explanation = explainer(processed_data, targets=[11, 14])

        assert len(explanation.saliency_map) == 2

        # Black-box
        model = ov.Core().read_model(model_path)
        explainer = Explainer(
            model,
            task=Task.CLASSIFICATION,
            explain_mode=ExplainMode.BLACKBOX,
            postprocess_fn=get_postprocess_fn(),
        )

        processed_data = self.preprocess_fn(self.image)
        explanation = explainer(processed_data, targets=[11, 14], preset=Preset.SPEED)

        assert len(explanation.saliency_map) == 2

    def test_auto_black_box_fallback(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        with pytest.raises(Exception) as exc_info:
            explainer = Explainer(
                model=model,
                task=Task.CLASSIFICATION,
                preprocess_fn=self.preprocess_fn,
                explain_mode=ExplainMode.AUTO,
                target_layer="some_wrong_name",
            )
        assert str(exc_info.value) == "Postprocess function has to be provided for the black-box mode."

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=ExplainMode.AUTO,
            target_layer="some_wrong_name",
        )
        explanation = explainer(self.image, targets=-1, num_iterations_per_kernel=2)
        assert len(explanation.saliency_map) == 20

    def test_two_explainers_with_same_model(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
        )

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
        )

    @pytest.mark.parametrize(
        "targets",
        [
            -1,
            3,
            [3],
            [1, 3],
            np.uint(1),
            np.int16(1),
            np.int64(1),
            np.array([1]),
            np.array([1, 2]),
        ],
    )
    def test_different_target_format(self, targets):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
        )

        explanation = explainer(
            self.image,
            targets=targets,
        )
        if isinstance(targets, int) and targets == -1:
            assert len(explanation.targets) == 20
        else:
            assert len(explanation.targets) == len(np.atleast_1d(np.asarray(targets)))

    def test_overlay_with_resize(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
        )

        explanation = explainer(
            self.image,
            targets=0,
            overlay=True,
            output_size=(50, 70),
        )
        assert explanation.saliency_map[0].shape == (50, 70, 3)

    def test_overlay_with_original_image(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
        )

        explanation = explainer(
            self.image,
            original_input_image=cv2.resize(src=self.image, dsize=(120, 100)),
            targets=0,
            overlay=True,
        )
        assert explanation.saliency_map[0].shape == (100, 120, 3)

    def test_init_with_file(self, mocker: MockerFixture):
        ov_core = mocker.patch("openvino.Core")
        create_method = mocker.patch("openvino_xai.Explainer.create_method")
        # None
        explainer = Explainer(
            model=None,
            task=Task.CLASSIFICATION,
        )
        ov_core.assert_not_called()
        # str
        path = "model.xml"
        explainer = Explainer(
            model=path,
            task=Task.CLASSIFICATION,
        )
        ov_core.return_value.read_model.assert_called_with(path)
        # Path
        path = Path("model.xml")
        explainer = Explainer(
            model=path,
            task=Task.CLASSIFICATION,
        )
        ov_core.return_value.read_model.assert_called_with(path)
