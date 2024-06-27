# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import shutil
from pathlib import Path
from time import time

import cv2
import numpy as np
import openvino.runtime as ov
import pandas as pd
import pytest

from openvino_xai.common.parameters import Method, Task
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
    get_score,
)
from openvino_xai.explainer.visualizer import Visualizer
from openvino_xai.utils.model_export import export_to_ir, export_to_onnx

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")


from tests.intg.test_classification_timm import (
    LIMITED_DIVERSE_SET_OF_CNN_MODELS,
    LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS,
    NON_SUPPORTED_BY_WB_MODELS,
)

TEST_MODELS = (
    LIMITED_DIVERSE_SET_OF_CNN_MODELS + LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS + NON_SUPPORTED_BY_WB_MODELS
)


def seed_everything(seed: int):
    """Set random seed."""
    import os
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class TestPerfClassificationTimm:
    clear_cache_converted_models = False
    clear_cache_hf_models = False
    supported_num_classes = {
        1000: 293,  # 293 is a cheetah class_id in the ImageNet-1k dataset
        21841: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        21843: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        11821: 1652,  # 1652 is a cheetah class_id in the ImageNet-12k dataset
    }

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root, fxt_output_root, fxt_clear_cache):
        self.data_dir = fxt_data_root
        self.output_dir = fxt_output_root
        self.cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        self.clear_cache_hf_models = fxt_clear_cache
        self.clear_cache_converted_models = fxt_clear_cache

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_white_box(self, model_id: str, fxt_num_repeat: int, fxt_tags: dict):
        if model_id in NON_SUPPORTED_BY_WB_MODELS:
            pytest.skip(reason="Not supported yet")

        timm_model, model_cfg = self.get_timm_model(model_id)

        ir_path = self.data_dir / "timm_models" / "converted_models" / model_id / "model_fp32.xml"
        if not ir_path.is_file():
            output_model_dir = self.output_dir / "timm_models" / "converted_models" / model_id
            output_model_dir.mkdir(parents=True, exist_ok=True)
            ir_path = output_model_dir / "model_fp32.xml"
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            set_dynamic_batch = model_id in LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS
            export_to_onnx(timm_model, onnx_path, dummy_tensor, set_dynamic_batch)
            export_to_ir(onnx_path, output_model_dir / "model_fp32.xml")

        if model_id in LIMITED_DIVERSE_SET_OF_CNN_MODELS:
            explain_method = Method.RECIPROCAM
        elif model_id in LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS:
            explain_method = Method.VITRECIPROCAM
        else:
            raise ValueError

        mean_values = [(item * 255) for item in model_cfg["mean"]]
        scale_values = [(item * 255) for item in model_cfg["std"]]
        preprocess_fn = get_preprocess_fn(
            change_channel_order=True,
            input_size=model_cfg["input_size"][1:],
            mean=mean_values,
            std=scale_values,
            hwc_to_chw=True,
        )

        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        image = cv2.imread("tests/assets/cheetah_person.jpg")

        records = []
        for seed in range(fxt_num_repeat):
            seed_everything(seed)

            record = fxt_tags.copy()
            record["model"] = model_id
            record["method"] = explain_method
            record["seed"] = seed

            model = ov.Core().read_model(ir_path)

            start_time = time()

            explainer = Explainer(
                model=model,
                task=Task.CLASSIFICATION,
                preprocess_fn=preprocess_fn,
                explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
                explain_method=explain_method,
                embed_scaling=False,
            )
            explanation = explainer(
                image,
                targets=[target_class],
                resize=True,
                colormap=True,
                overlay=True,
            )

            explain_time = time() - start_time
            record["time"] = explain_time

            assert explanation is not None
            assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
            print(record)
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / f"perf-raw-wb-{model_id}.csv")

        self.clear_cache()

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_black_box(self, model_id, fxt_num_repeat: int, fxt_num_masks: int, fxt_tags: dict):
        timm_model, model_cfg = self.get_timm_model(model_id)

        onnx_path = self.data_dir / "timm_models" / "converted_models" / model_id / "model_fp32.onnx"
        if not onnx_path.is_file():
            output_model_dir = self.output_dir / "timm_models" / "converted_models" / model_id
            output_model_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = output_model_dir / "model_fp32.onnx"
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            export_to_onnx(timm_model, onnx_path, dummy_tensor, False)

        model = ov.Core().read_model(onnx_path)

        mean_values = [(item * 255) for item in model_cfg["mean"]]
        scale_values = [(item * 255) for item in model_cfg["std"]]
        preprocess_fn = get_preprocess_fn(
            change_channel_order=True,
            input_size=model_cfg["input_size"][1:],
            mean=mean_values,
            std=scale_values,
            hwc_to_chw=True,
        )

        postprocess_fn = get_postprocess_fn()

        image = cv2.imread("tests/assets/cheetah_person.jpg")
        target_class = self.supported_num_classes[model_cfg["num_classes"]]

        records = []
        for seed in range(fxt_num_repeat):
            seed_everything(seed)

            record = fxt_tags.copy()
            record["model"] = model_id
            record["method"] = Method.RISE
            record["seed"] = seed
            record["num_masks"] = fxt_num_masks

            start_time = time()

            explainer = Explainer(
                model=model,
                task=Task.CLASSIFICATION,
                preprocess_fn=preprocess_fn,
                postprocess_fn=postprocess_fn,
                explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
            )
            explanation = explainer(
                image,
                targets=[target_class],
                resize=True,
                colormap=True,
                overlay=True,
                num_masks=fxt_num_masks,  # kwargs of the RISE algo
            )

            explain_time = time() - start_time
            record["time"] = explain_time

            assert explanation is not None
            assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
            print(record)
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / f"perf-raw-bb-{model_id}.csv", index=False)

        self.clear_cache()

    def get_timm_model(self, model_id):
        timm_model = timm.create_model(model_id, in_chans=3, pretrained=True, checkpoint_path="")
        timm_model.eval()
        model_cfg = timm_model.default_cfg
        num_classes = model_cfg["num_classes"]
        if num_classes not in self.supported_num_classes:
            self.clear_cache()
            pytest.skip(f"Number of model classes {num_classes} unknown")
        return timm_model, model_cfg

    def clear_cache(self):
        if self.clear_cache_converted_models:
            ir_model_dir = self.data_dir / "timm_models" / "converted_models"
            if ir_model_dir.is_dir():
                shutil.rmtree(ir_model_dir)
        if self.clear_cache_hf_models:
            huggingface_hub_dir = self.cache_dir / "huggingface" / "hub"
            if huggingface_hub_dir.is_dir():
                shutil.rmtree(huggingface_hub_dir)
