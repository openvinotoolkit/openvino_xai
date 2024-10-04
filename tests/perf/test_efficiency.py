# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path
from time import time

import cv2
import openvino as ov
import pandas as pd
import pytest

from openvino_xai.common.parameters import Method, Task
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.utils.model_export import export_to_onnx
from tests.perf.perf_tests_utils import (
    clear_cache,
    convert_timm_to_ir,
    get_timm_model,
    seed_everything,
)

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


class TestEfficiency:
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

        _, model_cfg = convert_timm_to_ir(model_id, self.data_dir, self.supported_num_classes)
        ir_path = self.data_dir / "timm_models" / "converted_models" / model_id / "model_fp32.xml"

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
        df.to_csv(self.output_dir / f"perf-raw-wb-{model_id}-{explain_method}.csv")

        clear_cache(self.data_dir, self.cache_dir, self.clear_cache_converted_models, self.clear_cache_hf_models)

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    @pytest.mark.parametrize("method", [Method.AISE, Method.RISE])
    def test_classification_black_box(
        self, model_id: str, method: Method, fxt_num_repeat: int, fxt_preset: str, fxt_tags: dict
    ):
        timm_model, model_cfg = get_timm_model(model_id, self.supported_num_classes)

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
            record["method"] = method
            record["seed"] = seed
            record["preset"] = fxt_preset

            start_time = time()

            explainer = Explainer(
                model=model,
                task=Task.CLASSIFICATION,
                preprocess_fn=preprocess_fn,
                postprocess_fn=postprocess_fn,
                explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
                explain_method=method,  # defaults to AISE
            )
            explanation = explainer(
                image,
                targets=[target_class],
                resize=True,
                colormap=True,
                overlay=True,
                preset=Preset(fxt_preset),  # kwargs of the black box algo
            )

            explain_time = time() - start_time
            record["time"] = explain_time

            assert explanation is not None
            assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
            print(record)
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / f"perf-raw-bb-{model_id}-{method}.csv", index=False)

        clear_cache(self.data_dir, self.cache_dir, self.clear_cache_converted_models, self.clear_cache_hf_models)
