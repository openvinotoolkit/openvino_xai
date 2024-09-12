# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai import Task, insert_xai
from openvino_xai.common.utils import logger, softmax

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")


TEST_MODELS = timm.list_models(pretrained=True)

NOT_SUPPORTED_BY_BB_MODELS = {
    "repvit": "urllib.error.HTTPError: HTTP Error 404: Not Found",
    "tf_efficientnet_cc": "torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of convolution for kernel of unknown shape.",
    "vit_base_r50_s16_224.orig_in21k": "RuntimeError: Error(s) in loading state_dict for VisionTransformer",
    "vit_gigantic_patch16_224_ijepa.in22k": "RuntimeError: shape '[1, 13, 13, -1]' is invalid for input of size 274560",
    "vit_huge_patch14_224.orig_in21k": "RuntimeError: Error(s) in loading state_dict for VisionTransformer",
    "vit_large_patch32_224.orig_in21k": "RuntimeError: Error(s) in loading state_dict for VisionTransformer",
    "volo_": "RuntimeError: Exception from src/core/src/dimension.cpp:227: Cannot get length of dynamic dimension",
}

SUPPORTED_BUT_FAILED_BY_WB_MODELS = {
    "swin": "Only two outputs of the between block Add node supported, but got 1. Try to use black-box.",
    "vit_base_patch16_rpn_224": "Number of normalization outputs > 1",
    "vit_relpos_medium_patch16_rpn_224": "ValueError in openvino_xai/methods/white_box/recipro_cam.py:215",
}

NOT_SUPPORTED_BY_WB_MODELS = {
    **NOT_SUPPORTED_BY_BB_MODELS,
    # Killed on WB
    "beit_large_patch16_512": "Failed to allocate 94652825600 bytes of memory",
    "convmixer_1536_20": "OOM Killed",
    "eva_large_patch14_336": "OOM Killed",
    "eva02_base_patch14_448": "OOM Killed",
    "eva02_large_patch14_448": "OOM Killed",
    "mobilevit_": "Segmentation fault",
    "mobilevit_xxs": "Segmentation fault",
    "mvitv2_base.fb_in1k": "Segmentation fault",
    "mvitv2_large": "OOM Killed",
    "mvitv2_small": "Segmentation fault",
    "mvitv2_tiny": "Segmentation fault",
    "pit_": "Segmentation fault",
    "pvt_": "Segmentation fault",
    "tf_efficientnet_l2.ns_jft_in1k": "OOM Killed",
    "xcit_large": "Failed to allocate 81581875200 bytes of memory",
    "xcit_medium_24_p8_384": "OOM Killed",
    "xcit_small_12_p8_384": "OOM Killed",
    "xcit_small_24_p8_384": "OOM Killed",
    # Not expected to work for now
    "cait_": "Cannot create an empty Constant. Please provide valid data.",
    "coat_": "Only two outputs of the between block Add node supported, but got 1.",
    "crossvit": "One (and only one) of the nodes has to be Add type. But got StridedSlice and StridedSlice.",
    # work in CNN mode -> "davit": "Only two outputs of the between block Add node supported, but got 1.",
    # work in CNN mode -> "efficientformer": "Cannot find output backbone_node in auto mode.",
    # work in CNN mode -> "focalnet": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    # work in CNN mode -> "gcvit": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "levit_": "Check 'TRShape::merge_into(output_shape, in_copy)' failed",
    # work in CNN mode -> "maxvit": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    # work in CNN mode -> "maxxvit": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    # work in CNN mode -> "mobilevitv2": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    # work in CNN mode -> "nest_": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    # work in CNN mode -> "poolformer": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "sequencer2d": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "tnt_s_patch16_224": "Only two outputs of the between block Add node supported, but got 1.",
    "twins": "One (and only one) of the nodes has to be Add type. But got ShapeOf and Transpose.",
    # work in CNN mode -> "visformer": "Cannot find output backbone_node in auto mode, please provide target_layer",
}


class TestTorchOnnxTimm:
    clear_cache_converted_models = False
    clear_cache_hf_models = False

    @pytest.fixture(autouse=True)
    def setup(self, fxt_clear_cache):
        self.clear_cache_hf_models = fxt_clear_cache
        self.clear_cache_converted_models = fxt_clear_cache

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    # @pytest.mark.parametrize("model_id", ["resnet18.a1_in1k"])
    def test_insert_xai(self, model_id, fxt_output_root: Path):
        # for skipped_model in NOT_SUPPORTED_BY_WB_MODELS.keys():
        #    if skipped_model in model_id:
        #        pytest.skip(reason=NOT_SUPPORTED_BY_WB_MODELS[skipped_model])

        # for failed_model in SUPPORTED_BUT_FAILED_BY_WB_MODELS.keys():
        #    if failed_model in model_id:
        #        pytest.xfail(reason=SUPPORTED_BUT_FAILED_BY_WB_MODELS[failed_model])

        # Load Torch model from timm
        model = timm.create_model(model_id, in_chans=3, pretrained=True)
        input_size = model.default_cfg["input_size"][1:]  # (H, W)
        input_mean = np.array(model.default_cfg["mean"])
        input_std = np.array(model.default_cfg["std"])

        # Load image
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        image = cv2.resize(image, dsize=input_size)
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        image_norm = ((image / 255.0 - input_mean) / input_std).astype(np.float32)
        image_norm = image_norm.transpose((2, 0, 1))  # HxWxC -> CxHxW
        image_norm = image_norm[None, :]  # CxHxW -> 1xCxHxW

        # Insert XAI head
        model_xai: torch.nn.Module = insert_xai(model, Task.CLASSIFICATION)

        # Torch XAI model inference
        model_xai.eval()
        with torch.no_grad():
            outputs = model_xai(torch.from_numpy(image_norm))
            logits = outputs["prediction"]  # BxC
            saliency_maps = outputs["saliency_map"]  # BxCxhxw
            probs = torch.softmax(logits, dim=-1)
            label = probs.argmax(dim=-1)[0]
        assert probs[0, label] > 0

        # Torch XAI model saliency map
        saliency_maps = saliency_maps.numpy(force=True).squeeze(0)  # Cxhxw
        saliency_map = saliency_maps[label]  # hxw mask for the label
        assert saliency_map.shape[-1] > 1 and saliency_map.shape[-2] > 1
        assert saliency_map.min() < saliency_map.max()
        assert saliency_map.dtype == np.uint8

        # ONNX model conversion
        model_path = fxt_output_root / "func" / "onnx" / "model.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model_xai,
            torch.from_numpy(image_norm),
            model_path,
            input_names=["input"],
            output_names=["prediction", "saliency_map"],
        )
        assert model_path.exists()

        # ONNX model inference
        session = onnxruntime.InferenceSession(model_path)
        outputs = session.run(
            output_names=["prediction", "saliency_map"],
            input_feed={"input": image_norm.astype(np.float32)},
        )
        logits, saliency_maps = outputs  # NOTE: dict keys are removed in Torch->ONNX conversion
        probs = softmax(logits)
        label = probs.argmax(axis=-1)[0]
        assert probs[0, label] > 0

        # ONNX XAI model saliency map
        saliency_maps = saliency_maps.squeeze(0)  # Cxhxw
        saliency_map = saliency_maps[label]  # hxw mask for the label
        assert saliency_map.shape[-1] > 1 and saliency_map.shape[-2] > 1
        assert saliency_map.min() < saliency_map.max()
        assert saliency_map.dtype == np.uint8

        # Clean up
        self.clear_cache()

    def clear_cache(self):
        if self.clear_cache_converted_models:
            ir_model_dir = self.output_dir / "timm_models" / "converted_models"
            if ir_model_dir.is_dir():
                shutil.rmtree(ir_model_dir)
        if self.clear_cache_hf_models:
            cache_dir = os.environ.get("XDG_CACHE_HOME", "~/.cache")
            huggingface_hub_dir = Path(cache_dir) / "huggingface/hub/"
            if huggingface_hub_dir.is_dir():
                shutil.rmtree(huggingface_hub_dir)
