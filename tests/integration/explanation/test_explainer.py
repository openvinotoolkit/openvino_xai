# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pytest

import openvino.runtime as ov
import cv2

from openvino_xai.common.parameters import TaskType
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explanation import ExplanationParameters, TargetExplainGroup, ExplainMode
from openvino_xai.explanation.explainers import Explainer
from openvino_xai.explanation.utils import get_preprocess_fn, get_postprocess_fn
from openvino_xai.insertion import ClassificationInsertionParameters

MODEL_NAME = "mlc_mobilenetv3_large_voc"


class TestExplainer:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    data_dir = Path(".data")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )

    @pytest.mark.parametrize(
        "explain_mode",
        [
            ExplainMode.WHITEBOX,
            ExplainMode.BLACKBOX,
            ExplainMode.AUTO,
        ],
    )
    @pytest.mark.parametrize(
        "target_explain_group",
        [
            TargetExplainGroup.ALL,
            TargetExplainGroup.CUSTOM,
        ],
    )
    def test_explainer(self, explain_mode, target_explain_group):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explainer = Explainer(
            model,
            task_type=TaskType.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=explain_mode,
        )

        voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        explanation_parameters = ExplanationParameters(
            target_explain_group=target_explain_group,
            target_explain_labels=[11, 14],
            label_names=voc_labels,  # optional
        )
        explanation = explainer(self.image, explanation_parameters, num_masks=10)

        if target_explain_group == TargetExplainGroup.ALL:
            assert len(explanation.saliency_map) == 20
        if target_explain_group == TargetExplainGroup.CUSTOM:
            assert len(explanation.saliency_map) == 2

    def test_auto_black_box_fallback(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        with pytest.raises(Exception) as exc_info:
            insertion_parameters = ClassificationInsertionParameters(
                target_layer="some_wrong_name",
            )
            explainer = Explainer(
                model=model,
                task_type=TaskType.CLASSIFICATION,
                preprocess_fn=self.preprocess_fn,
                explain_mode=ExplainMode.AUTO,
                insertion_parameters=insertion_parameters,
            )
            assert explainer.explain_mode == ExplainMode.BLACKBOX
        assert str(exc_info.value) == "Postprocess function has to be provided for the black-box mode."

        insertion_parameters = ClassificationInsertionParameters(
            target_layer="some_wrong_name",
        )
        explainer = Explainer(
            model=model,
            task_type=TaskType.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=ExplainMode.AUTO,
            insertion_parameters=insertion_parameters,
        )
        assert explainer.explain_mode == ExplainMode.BLACKBOX
        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.ALL,
        )
        explanation = explainer(self.image, explanation_parameters, num_masks=10)
        assert len(explanation.saliency_map) == 20
