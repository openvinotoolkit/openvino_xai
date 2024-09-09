# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import openvino as ov
import pytest
import torch
from pytest_mock import MockerFixture

from openvino_xai.common.parameters import Method, Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.aise.classification import AISEClassification
from openvino_xai.methods.factory import BlackBoxMethodFactory, WhiteBoxMethodFactory
from openvino_xai.methods.white_box import torch as torch_method
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
    explain_method = WhiteBoxMethodFactory.create_method(Task.CLASSIFICATION, model_cnn, PREPROCESS_FN)
    assert isinstance(explain_method, ReciproCAM)

    model_cnn = ov.Core().read_model(model_path)
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, explain_method=Method.RECIPROCAM
    )
    assert isinstance(explain_method, ReciproCAM)

    model_cnn = ov.Core().read_model(model_path)
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_cnn, PREPROCESS_FN, explain_method=Method.ACTIVATIONMAP
    )
    assert isinstance(explain_method, ActivationMap)

    model_cnn = ov.Core().read_model(model_path)
    with pytest.raises(Exception) as exc_info:
        explain_method = WhiteBoxMethodFactory.create_method(
            Task.CLASSIFICATION,
            model_cnn,
            PREPROCESS_FN,
            explain_method="abc",
        )
    assert str(exc_info.value) == "Requested explanation method abc is not implemented."


def test_create_wb_cls_vit_method(fxt_data_root: Path):
    retrieve_otx_model(fxt_data_root, VIT_MODEL)
    model_path = fxt_data_root / "otx_models" / (VIT_MODEL + ".xml")
    model_vit = ov.Core().read_model(model_path)
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model_vit, PREPROCESS_FN, explain_method=Method.VITRECIPROCAM
    )
    assert isinstance(explain_method, ViTReciproCAM)


def test_create_wb_cls_guess_method(mocker: MockerFixture):
    model = mocker.MagicMock()
    # method=None -> ReciproCAM fail -> ViTReciproCAM
    recipro_cam = mocker.patch("openvino_xai.methods.factory.ReciproCAM", side_effect=Exception("DUMMY REASON"))
    vit_recipro_cam = mocker.patch("openvino_xai.methods.factory.ViTReciproCAM")
    explain_method = WhiteBoxMethodFactory.create_method(
        task=Task.CLASSIFICATION,
        model=model,
        preprocess_fn=PREPROCESS_FN,
        explain_method=None,
    )
    vit_recipro_cam.assert_called()
    # method=ReciproCAM -> ReciproCAM fail -> Exception
    recipro_cam = mocker.patch("openvino_xai.methods.factory.ReciproCAM", side_effect=Exception("DUMMY REASON"))
    vit_recipro_cam = mocker.patch("openvino_xai.methods.factory.ViTReciproCAM")
    with pytest.raises(Exception) as exc_info:
        explain_method = WhiteBoxMethodFactory.create_method(
            task=Task.CLASSIFICATION,
            model=model,
            preprocess_fn=PREPROCESS_FN,
            explain_method=Method.RECIPROCAM,
        )
    vit_recipro_cam.assert_not_called()
    assert str(exc_info.value) == "DUMMY REASON"


def test_create_bb_cls_vit_method(fxt_data_root: Path):
    retrieve_otx_model(fxt_data_root, VIT_MODEL)
    model_path = fxt_data_root / "otx_models" / (VIT_MODEL + ".xml")
    model_vit = ov.Core().read_model(model_path)
    explain_method = BlackBoxMethodFactory.create_method(Task.CLASSIFICATION, model_vit, get_postprocess_fn())
    assert isinstance(explain_method, AISEClassification)


def test_create_wb_det_cnn_method(fxt_data_root: Path):
    retrieve_otx_model(fxt_data_root, DEFAULT_DET_MODEL)
    model_path = fxt_data_root / "otx_models" / (DEFAULT_DET_MODEL + ".xml")
    model = ov.Core().read_model(model_path)

    sal_map_size = (23, 23)
    cls_head_output_node_names = MODEL_CONFIGS[DEFAULT_DET_MODEL].node_names

    explain_method = WhiteBoxMethodFactory.create_method(
        Task.DETECTION,
        model,
        PREPROCESS_FN,
        target_layer=cls_head_output_node_names,
        explain_method=Method.DETCLASSPROBABILITYMAP,
        num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
        saliency_map_size=sal_map_size,
    )
    assert isinstance(explain_method, DetClassProbabilityMap)

    model = ov.Core().read_model(model_path)
    with pytest.raises(Exception) as exc_info:
        explain_method = WhiteBoxMethodFactory.create_method(Task.DETECTION, model, PREPROCESS_FN, target_layer=None)
    assert str(exc_info.value) == "target_layer is required for the detection."

    model = ov.Core().read_model(model_path)
    with pytest.raises(Exception) as exc_info:
        explain_method = WhiteBoxMethodFactory.create_method(
            Task.DETECTION,
            model,
            PREPROCESS_FN,
            target_layer=cls_head_output_node_names,
            explain_method="abc",
            num_anchors=MODEL_CONFIGS[DEFAULT_DET_MODEL].anchors,
            saliency_map_size=sal_map_size,
        )
    assert str(exc_info.value) == "Requested explanation method abc is not implemented."


def test_create_torch_method(mocker: MockerFixture):
    model = {}
    with pytest.raises(ValueError):
        explain_method = BlackBoxMethodFactory.create_method(Task.CLASSIFICATION, model, get_postprocess_fn())
    model = torch.nn.Module()
    with pytest.raises(NotImplementedError):
        explain_method = BlackBoxMethodFactory.create_method(Task.CLASSIFICATION, model, get_postprocess_fn())
    with pytest.raises(NotImplementedError):
        explain_method = BlackBoxMethodFactory.create_method(
            Task.DETECTION, model, get_postprocess_fn(), target_layer=""
        )

    model = {}
    with pytest.raises(ValueError):
        explain_method = WhiteBoxMethodFactory.create_method(Task.CLASSIFICATION, model, get_postprocess_fn())
    model = torch.nn.Module()
    with pytest.raises(NotImplementedError):
        explain_method = WhiteBoxMethodFactory.create_method(
            Task.DETECTION, model, get_postprocess_fn(), target_layer=""
        )

    mocker.patch.object(torch_method.TorchActivationMap, "prepare_model")
    mocker.patch.object(torch_method.TorchReciproCAM, "prepare_model")
    mocker.patch.object(torch_method.TorchViTReciproCAM, "prepare_model")

    model = torch.nn.Module()
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model, get_postprocess_fn(), explain_method=Method.ACTIVATIONMAP
    )
    assert isinstance(explain_method, torch_method.TorchActivationMap)
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model, get_postprocess_fn(), explain_method=Method.RECIPROCAM
    )
    assert isinstance(explain_method, torch_method.TorchReciproCAM)
    explain_method = WhiteBoxMethodFactory.create_method(
        Task.CLASSIFICATION, model, get_postprocess_fn(), explain_method=Method.VITRECIPROCAM
    )
    assert isinstance(explain_method, torch_method.TorchViTReciproCAM)
