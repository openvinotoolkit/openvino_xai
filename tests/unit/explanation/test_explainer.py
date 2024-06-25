# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import openvino.runtime as ov
import pytest

from openvino_xai.api.api import insert_xai
from openvino_xai.common.parameters import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

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
    @pytest.mark.parametrize(
        "explain_all_classes",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize("with_xai_originally", [True, False])
    def test_explainer(self, explain_mode, explain_all_classes, with_xai_originally):
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

        if explain_all_classes:
            explanation = explainer(
                self.image,
                targets=-1,
                label_names=VOC_NAMES,  # optional
                num_masks=10,
            )
        else:
            explanation = explainer(
                self.image,
                targets=[11, 14],
                label_names=VOC_NAMES,  # optional
                num_masks=10,
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
        explanation = explainer(processed_data, targets=[11, 14], num_masks=10)

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
        explanation = explainer(self.image, targets=-1, num_masks=10)
        assert len(explanation.saliency_map) == 20
