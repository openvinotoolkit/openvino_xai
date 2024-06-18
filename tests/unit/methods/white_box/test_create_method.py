# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import openvino.runtime as ov
import pytest

from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.utils import get_preprocess_fn
from openvino_xai.inserter.parameters import (
    ClassificationInsertionParameters,
    DetectionInsertionParameters,
)
from openvino_xai.methods.factory import WhiteBoxMethodFactory
from openvino_xai.methods.white_box.activation_map import ActivationMap
from openvino_xai.methods.white_box.det_class_probability_map import (
    DetClassProbabilityMap,
)
from openvino_xai.methods.white_box.recipro_cam import ReciproCAM, ViTReciproCAM
from tests.intg.test_classification import DEFAULT_CLS_MODEL
from tests.intg.test_detection import DEFAULT_DET_MODEL, MODEL_CONFIGS

VIT_MODEL = "deit-tiny"

PREPROCESS_FN = get_preprocess_fn(
    change_channel_order=True,
    input_size=(224, 224),
    hwc_to_chw=True,
)


def test_create_wb_cls_cnn_method(fxt_data_root: Path):
    retrieve_otx_model(fxt_data_root, DEFAULT_CLS_MODEL)
    model_path = fxt_data_root / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")

    model_cnn = ov.Core().read_model(model_path)
    insertion_parameters = None
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, insertion_parameters
    )
    assert isinstance(explain_method, ReciproCAM)

    model_cnn = ov.Core().read_model(model_path)
    insertion_parameters = ClassificationInsertionParameters()
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, insertion_parameters
    )
    assert isinstance(explain_method, ReciproCAM)

    model_cnn = ov.Core().read_model(model_path)
    insertion_parameters = ClassificationInsertionParameters(
        explain_method=Method.RECIPROCAM,
    )
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, insertion_parameters
    )
    assert isinstance(explain_method, ReciproCAM)

    model_cnn = ov.Core().read_model(model_path)
    insertion_parameters = ClassificationInsertionParameters(
        explain_method=Method.ACTIVATIONMAP,
    )
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, insertion_parameters
    )
    assert isinstance(explain_method, ActivationMap)

    model_cnn = ov.Core().read_model(model_path)
    with pytest.raises(Exception) as exc_info:
        insertion_parameters = ClassificationInsertionParameters(
            explain_method="abc",
        )
        explain_method = WhiteBoxMethodFactory.create_method(
            Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, insertion_parameters
        )
    assert str(exc_info.value) == "Requested explanation method abc is not implemented."


def test_create_wb_cls_vit_method(fxt_data_root: Path):
    retrieve_otx_model(fxt_data_root, VIT_MODEL)
    model_path = fxt_data_root / "otx_models" / (VIT_MODEL + ".xml")
    model_vit = ov.Core().read_model(model_path)
    insertion_parameters = ClassificationInsertionParameters(
        explain_method=Method.VITRECIPROCAM,
    )
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_vit, PREPROCESS_FN, insertion_parameters
    )
    assert isinstance(explain_method, ViTReciproCAM)


def test_create_wb_det_cnn_method(fxt_data_root: Path):
    retrieve_otx_model(fxt_data_root, DEFAULT_DET_MODEL)
    model_path = fxt_data_root / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
    model = ov.Core().read_model(model_path)

    sal_map_size = (23, 23)
    cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names
    insertion_parameters = DetectionInsertionParameters(
        target_layer=cls_head_output_node_names,
        num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
        saliency_map_size=sal_map_size,
        explain_method=Method.DETCLASSPROBABILITYMAP,
    )

    explain_method = WhiteBoxMethodFactory.create_method(Task.DETECTION, model, PREPROCESS_FN, insertion_parameters)
    assert isinstance(explain_method, DetClassProbabilityMap)

    model = ov.Core().read_model(model_path)
    with pytest.raises(Exception) as exc_info:
        explain_method = WhiteBoxMethodFactory.create_method(
            Task.DETECTION, model, PREPROCESS_FN, insertion_parameters=None
        )
    assert str(exc_info.value) == "insertion_parameters is required for the detection models."

    model = ov.Core().read_model(model_path)
    with pytest.raises(Exception) as exc_info:
        insertion_parameters = DetectionInsertionParameters(
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
            saliency_map_size=sal_map_size,
            explain_method="abc",
        )
        explain_method = WhiteBoxMethodFactory.create_method(Task.DETECTION, model, PREPROCESS_FN, insertion_parameters)
    assert str(exc_info.value) == "Requested explanation method abc is not implemented."
