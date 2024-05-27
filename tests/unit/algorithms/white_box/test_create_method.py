# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import openvino.runtime as ov
import pytest

from openvino_xai.algorithms.white_box.create_method import (
    create_white_box_classification_explain_method,
    create_white_box_detection_explain_method,
)
from openvino_xai.algorithms.white_box.white_box_methods import (
    ActivationMapXAIMethod,
    DetClassProbabilityMapXAIMethod,
    ReciproCAMXAIMethod,
    ViTReciproCAMXAIMethod,
)
from openvino_xai.common.parameters import XAIMethodType
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.insertion.insertion_parameters import (
    ClassificationInsertionParameters,
    DetectionInsertionParameters,
)
from tests.integration.test_classification import DEFAULT_CLS_MODEL
from tests.integration.test_detection import DEFAULT_DET_MODEL, MODEL_CONFIGS

DATA_DIR = Path(".data")
VIT_MODEL = "deit-tiny"


def test_create_wb_cls_cnn_method():
    retrieve_otx_model(DATA_DIR, DEFAULT_CLS_MODEL)
    model_path = DATA_DIR / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
    model_cnn = ov.Core().read_model(model_path)

    insertion_parameters = None
    explain_method = create_white_box_classification_explain_method(model_cnn, insertion_parameters)
    assert isinstance(explain_method, ReciproCAMXAIMethod)

    insertion_parameters = ClassificationInsertionParameters()
    explain_method = create_white_box_classification_explain_method(model_cnn, insertion_parameters)
    assert isinstance(explain_method, ReciproCAMXAIMethod)

    insertion_parameters = ClassificationInsertionParameters(
        explain_method_type=XAIMethodType.RECIPROCAM,
    )
    explain_method = create_white_box_classification_explain_method(model_cnn, insertion_parameters)
    assert isinstance(explain_method, ReciproCAMXAIMethod)

    insertion_parameters = ClassificationInsertionParameters(
        explain_method_type=XAIMethodType.ACTIVATIONMAP,
    )
    explain_method = create_white_box_classification_explain_method(model_cnn, insertion_parameters)
    assert isinstance(explain_method, ActivationMapXAIMethod)

    with pytest.raises(Exception) as exc_info:
        insertion_parameters = ClassificationInsertionParameters(
            explain_method_type="abc",
        )
        explain_method = create_white_box_classification_explain_method(model_cnn, insertion_parameters)
    assert str(exc_info.value) == "Requested explanation method abc is not implemented."


def test_create_wb_cls_vit_method():
    retrieve_otx_model(DATA_DIR, VIT_MODEL)
    model_path = DATA_DIR / "otx_models" / (VIT_MODEL + ".xml")
    model_vit = ov.Core().read_model(model_path)
    insertion_parameters = ClassificationInsertionParameters(
        explain_method_type=XAIMethodType.VITRECIPROCAM,
    )
    explain_method = create_white_box_classification_explain_method(model_vit, insertion_parameters)
    assert isinstance(explain_method, ViTReciproCAMXAIMethod)


def test_create_wb_det_cnn_method():
    retrieve_otx_model(DATA_DIR, DEFAULT_DET_MODEL)
    model_path = DATA_DIR / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
    model = ov.Core().read_model(model_path)

    sal_map_size = (23, 23)
    cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names
    insertion_parameters = DetectionInsertionParameters(
        target_layer=cls_head_output_node_names,
        num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
        saliency_map_size=sal_map_size,
        explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,
    )

    explain_method = create_white_box_detection_explain_method(model, insertion_parameters)
    assert isinstance(explain_method, DetClassProbabilityMapXAIMethod)

    with pytest.raises(Exception) as exc_info:
        explain_method = create_white_box_detection_explain_method(model, insertion_parameters=None)
    assert str(exc_info.value) == "insertion_parameters is required for the detection models."

    with pytest.raises(Exception) as exc_info:
        insertion_parameters = DetectionInsertionParameters(
            target_layer=cls_head_output_node_names,
            num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
            saliency_map_size=sal_map_size,
            explain_method_type="abc",
        )
        explain_method = create_white_box_detection_explain_method(model, insertion_parameters)
    assert str(exc_info.value) == "Requested explanation method abc is not implemented."
