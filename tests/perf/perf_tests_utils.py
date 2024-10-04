# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from typing import Dict

import pytest

from openvino_xai.utils.model_export import export_to_ir, export_to_onnx

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")


from tests.intg.test_classification_timm import (
    LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS,
)


def seed_everything(seed: int):
    """Set random seed."""
    import os
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def convert_timm_to_ir(model_id: str, data_dir: Path, supported_num_classes: Dict[int, int]):
    timm_model, model_cfg = get_timm_model(model_id, supported_num_classes)

    ir_path = data_dir / "timm_models" / "converted_models" / model_id / "model_fp32.xml"
    if not ir_path.is_file():
        output_model_dir = data_dir / "timm_models" / "converted_models" / model_id
        output_model_dir.mkdir(parents=True, exist_ok=True)
        ir_path = output_model_dir / "model_fp32.xml"
        input_size = [1] + list(model_cfg["input_size"])
        dummy_tensor = torch.rand(input_size)
        onnx_path = output_model_dir / "model_fp32.onnx"
        set_dynamic_batch = model_id in LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS
        export_to_onnx(timm_model, onnx_path, dummy_tensor, set_dynamic_batch)
        export_to_ir(onnx_path, output_model_dir / "model_fp32.xml")

    return timm_model, model_cfg


def get_timm_model(model_id: str, supported_num_classes: Dict[int, int]):
    timm_model = timm.create_model(model_id, in_chans=3, pretrained=True, checkpoint_path="")
    timm_model.eval()
    model_cfg = timm_model.default_cfg
    num_classes = model_cfg["num_classes"]
    if num_classes not in supported_num_classes:
        clear_cache()
        pytest.skip(f"Number of model classes {num_classes} unknown")
    return timm_model, model_cfg


def clear_cache(
    data_dir: Path, cache_dir: Path, clear_cache_converted_models: bool = False, clear_cache_hf_models: bool = False
):
    if clear_cache_converted_models:
        ir_model_dir = data_dir / "timm_models" / "converted_models"
        if ir_model_dir.is_dir():
            shutil.rmtree(ir_model_dir)
    if clear_cache_hf_models:
        huggingface_hub_dir = cache_dir / "huggingface" / "hub"
        if huggingface_hub_dir.is_dir():
            shutil.rmtree(huggingface_hub_dir)
