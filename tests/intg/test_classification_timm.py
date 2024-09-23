# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import shutil
import subprocess  # nosec B404 (not a part of product)
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai import insert_xai
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


LIMITED_DIVERSE_SET_OF_CNN_MODELS = [
    # Only 60K+ downloads in https://huggingface.co/timm?sort_models=downloads#models
    # "bat_resnext26ts.ch_in1k",
    "resnet18.a1_in1k",
    # "mobilenetv3_large_100.ra_in1k",
    # "tf_efficientnet_b0.aa_in1k",
    "botnet26t_256.c1_in1k",
    "convnext_base.clip_laion2b_augreg_ft_in1k",
    # "convnextv2_pico.fcmae_ft_in1k",
    "cs3darknet_l.c2ns_in1k",
    # "darknet53.c2ns_in1k",
    # "densenet121.ra_in1k",
    # "dla34.in1k",
    # "dpn68.mx_in1k",
    # "eca_botnext26ts_256.c1_in1k",
    # "ecaresnet26t.ra2_in1k",
    # "edgenext_base.in21k_ft_in1k",
    # "efficientnet_b0.ra_in1k",
    "ese_vovnet19b_dw.ra_in1k",
    "fbnetv3_b.ra2_in1k",
    "gernet_s.idstcv_in1k",
    # "hardcorenas_a.miil_green_in1k",
    "hrnet_w18.ms_aug_in1k",
    "inception_v3.gluon_in1k",
    "lcnet_050.ra2_in1k",
    # "legacy_senet154.in1k",
    "mixer_b16_224.goog_in21k",
    # "mixnet_s.ft_in1k",
    # "mobilenetv2_100.ra_in1k",
    "regnety_002.pycls_in1k",
    "repvgg_a2.rvgg_in1k",
    # "repvit_m1.dist_in1k",  # 404 Not Found
    # "res2net50_14w_8s.in1k",
    # "resmlp_12_224.fb_dino",
    # "resnetaa50.a1h_in1k",
    # "resnetrs50.tf_in1k",
    # "resnext26ts.ra2_in1k",
    "rexnet_100.nav_in1k",
    "selecsls42b.in1k",
    # "seresnet50.a1_in1k",
    # "seresnext26d_32x4d.bt_in1k",
    "tf_mixnet_l.in1k",
    # "tf_mobilenetv3_large_075.in1k",
    "tinynet_a.in1k",
    # "wide_resnet50_2.racm_in1k",
    # "xception41.tf_in1k",
    # "vgg11.tv_in1k",
    # "coatnet_0_rw_224.sw_in1k",
    # "focalnet_base_lrf.ms_in1k",
]


LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS = [
    "beit_base_patch16_224.in22k_ft_in22k_in1k",  # Downloads last month 41,778
    # "beit_large_patch16_224.in22k_ft_in22k_in1k",
    "deit_tiny_patch16_224.fb_in1k",  # Downloads last month 3,371
    # "deit_small_distilled_patch16_224.fb_in1k",
    # "deit_base_patch16_224.fb_in1k",
    "vit_tiny_patch16_224.augreg_in21k",  # Downloads last month 15,345
    # "vit_small_patch16_224.augreg_in1k",
    # "vit_base_patch8_224.augreg2_in21k_ft_in1k",
    # "vit_base_patch16_224.augreg2_in21k_ft_in1k",  # Downloads last month 161,508
    # "vit_base_patch32_224.augreg_in1k",
    # "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k",
    "convit_tiny.fb_in1k",
    "flexivit_small.300ep_in1k",
]


NON_SUPPORTED_BY_WB_MODELS = [
    # CNN, dynamic batch issue
    "nest_tiny_jx.goog_in1k",
    "pit_s_224.in1k",
    "pvt_v2_b0.in1k",
    "sequencer2d_l.in1k",
    "mobilevitv2_050.cvnets_in1k",
    # Transformer, various issues
    "convformer_b36.sail_in1k",
    "davit_tiny.msft_in1k",
    "poolformer_m36.sail_in1k",
    "caformer_b36.sail_in1k",
    "cait_m36_384.fb_dist_in1k",
    "coat_lite_mini.in1k",
    "crossvit_9_240.in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "swinv2_tiny_window8_256.ms_in1k",
    "twins_svt_small.in1k",
    "efficientformer_l1.snap_dist_in1k",
    "gcvit_tiny.in1k",
    "levit_128.fb_dist_in1k",
    "maxvit_base_tf_224.in1k",
    "mvitv2_base.fb_in1k",
    "poolformer_m36.sail_in1k",
    "xcit_nano_12_p8_224.fb_dist_in1k",
    "convmixer_768_32.in1k",
]


TEST_MODELS = (
    LIMITED_DIVERSE_SET_OF_CNN_MODELS + LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS + NON_SUPPORTED_BY_WB_MODELS
)


class TestImageClassificationTimm:
    fields = ["Model", "Exported to ONNX", "Exported to OV IR", "Explained", "Map size", "Map saved"]
    counter_row = ["Counters", "0", "0", "0", "-", "-"]
    report = [fields, counter_row]
    clear_cache_converted_models = False
    clear_cache_hf_models = False
    supported_num_classes = {
        1000: 293,  # 293 is a cheetah class_id in the ImageNet-1k dataset
        21841: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        21843: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        11821: 1652,  # 1652 is a cheetah class_id in the ImageNet-12k dataset
    }
    reference_maps_names = {
        (ExplainMode.WHITEBOX, Method.RECIPROCAM): Path("resnet18.a1_in1k_reciprocam.npy"),
        (ExplainMode.WHITEBOX, Method.ACTIVATIONMAP): Path("resnet18.a1_in1k_activationmap.npy"),
        (ExplainMode.BLACKBOX, Method.AISE): Path("resnet18.a1_in1k_aise.npy"),
        (ExplainMode.BLACKBOX, Method.RISE): Path("resnet18.a1_in1k_rise.npy"),
    }

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root, fxt_output_root, fxt_clear_cache):
        self.data_dir = fxt_data_root
        self.output_dir = fxt_output_root
        self.clear_cache_hf_models = fxt_clear_cache
        self.clear_cache_converted_models = fxt_clear_cache

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_white_box(self, model_id, dump_maps=False):
        # self.check_for_saved_map(model_id, "timm_models/maps_wb/")

        if model_id in NON_SUPPORTED_BY_WB_MODELS:
            pytest.skip(reason="Not supported yet")

        model_dir = self.data_dir / "timm_models" / "converted_models"
        timm_model, model_cfg = self.get_timm_model(model_id, model_dir)
        self.update_report("report_wb.csv", model_id)

        ir_path = model_dir / model_id / "model_fp32.xml"
        model = ov.Core().read_model(ir_path)

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

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
            explain_method=explain_method,
            embed_scaling=False,
        )

        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        explanation = explainer(
            image,
            targets=[target_class],
            resize=False,
            colormap=False,
        )

        assert explanation is not None
        assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.shape}.")
        self.update_report("report_wb.csv", model_id, "True", "True", "True")
        raw_shape = explanation.shape
        shape_str = "H=" + str(raw_shape[0]) + ", W=" + str(raw_shape[1])
        self.update_report("report_wb.csv", model_id, "True", "True", "True", shape_str)

        if dump_maps:
            # timm workaround to remove outlier activations at corners
            # TODO: find a root cause
            raw_sal_map = explanation.saliency_map[target_class]
            raw_sal_map[0, 0] = np.mean(np.delete(raw_sal_map[:2, :2].flatten(), 0))
            raw_sal_map[0, -1] = np.mean(np.delete(raw_sal_map[:2, -2:].flatten(), 1))
            raw_sal_map[-1, 0] = np.mean(np.delete(raw_sal_map[-2:, :2].flatten(), 2))
            raw_sal_map[-1, -1] = np.mean(np.delete(raw_sal_map[-2:, -2:].flatten(), 3))
            explanation.saliency_map[target_class] = raw_sal_map
            visualizer = Visualizer()
            explanation = visualizer(
                explanation=explanation,
                original_input_image=image,
                scaling=True,
                overlay=True,
                resize=False,
                colormap=False,
            )

            model_output = explainer.model_forward(image)
            target_confidence = get_score(model_output["logits"], target_class, activation=ActivationType.SOFTMAX)
            self.put_confidence_into_map_overlay(explanation, target_confidence, target_class)

            save_dir = self.output_dir / "timm_models" / "maps_wb"
            explanation.save(save_dir, model_id)
            file_name = model_id + "_target_" + str(target_class) + ".jpg"
            map_saved = (save_dir / file_name).is_file()
            self.update_report("report_wb.csv", model_id, "True", "True", "True", shape_str, str(map_saved))
        self.clear_cache()

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_black_box(self, model_id, dump_maps=False):
        # self.check_for_saved_map(model_id, "timm_models/maps_bb/")
        if model_id == "nest_tiny_jx.goog_in1k":
            pytest.xfail(
                "[cpu]reshape: the shape of input data (1.27.27.192) conflicts with the reshape pattern (0.2.14.2.14.192)"
            )

        model_dir = self.data_dir / "timm_models" / "converted_models"
        timm_model, model_cfg = self.get_timm_model(model_id, model_dir)
        self.update_report("report_bb.csv", model_id)

        ir_path = model_dir / model_id / "model_fp32.xml"
        model = ov.Core().read_model(ir_path)

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

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
        )

        image = cv2.imread("tests/assets/cheetah_person.jpg")
        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        explanation = explainer(
            image,
            targets=[target_class],
            overlay=True,
            num_iterations_per_kernel=2,
            kernel_widths=[0.1],
        )

        assert explanation is not None
        assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.shape}.")
        self.update_report("report_bb.csv", model_id, "True", "True", "True")
        shape = explanation.shape
        shape_str = "H=" + str(shape[0]) + ", W=" + str(shape[1])
        self.update_report("report_bb.csv", model_id, "True", "True", "True", shape_str)

        if dump_maps:
            model_output = explainer.model_forward(image)
            target_confidence = get_score(model_output["logits"], target_class, activation=ActivationType.SOFTMAX)
            self.put_confidence_into_map_overlay(explanation, target_confidence, target_class)

            save_dir = self.output_dir / "timm_models" / "maps_bb"
            explanation.save(save_dir, model_id)
            file_name = model_id + "_target_" + str(target_class) + ".jpg"
            map_saved = (save_dir / file_name).is_file()
            self.update_report("report_bb.csv", model_id, "True", "True", "True", shape_str, str(map_saved))
        self.clear_cache()

    @pytest.mark.parametrize(
        "model_id",
        [
            "resnet18.a1_in1k",
            "vit_tiny_patch16_224.augreg_in21k",  # Downloads last month 15,345
        ],
    )
    # @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_ovc_model_white_box(self, model_id):
        if model_id in NON_SUPPORTED_BY_WB_MODELS:
            pytest.skip(reason="Not supported yet")

        if "convit_tiny.fb_in1k" in model_id:
            pytest.skip(
                reason="RuntimeError: Couldn't get TorchScript module by tracing."
            )  # Torch -> OV conversion error

        model_dir = self.data_dir / "timm_models" / "converted_models"
        timm_model, model_cfg = self.get_timm_model(model_id, model_dir)
        input_size = list(timm_model.default_cfg["input_size"])
        dummy_tensor = torch.rand([1] + input_size)
        model = ov.convert_model(timm_model, example_input=dummy_tensor, input=(ov.PartialShape([-1] + input_size),))

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

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
            explain_method=explain_method,
            embed_scaling=False,
        )

        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        explanation = explainer(
            image,
            targets=[target_class],
            resize=False,
            colormap=False,
        )

        assert explanation is not None
        assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.shape}.")
        self.clear_cache()

    @pytest.mark.parametrize(
        "model_id",
        [
            "resnet18.a1_in1k",
            "vit_tiny_patch16_224.augreg_in21k",  # Downloads last month 15,345
        ],
    )
    @pytest.mark.parametrize(
        "explain_mode",
        [
            ExplainMode.WHITEBOX,
            ExplainMode.BLACKBOX,
        ],
    )
    @pytest.mark.parametrize(
        "model_format",
        ["xml", "onnx"],
    )
    def test_model_format(self, model_id, explain_mode, model_format):
        if (
            model_id == "vit_tiny_patch16_224.augreg_in21k"
            and explain_mode == ExplainMode.WHITEBOX
            and model_format == "onnx"
        ):
            pytest.xfail(
                "RuntimeError: Failed to insert XAI into the model -> Only two outputs of the between block Add node supported, but got 3."
            )

        model_dir = self.data_dir / "timm_models" / "converted_models"
        timm_model, model_cfg = self.get_timm_model(model_id, model_dir)
        model_path = model_dir / model_id / ("model_fp32." + model_format)

        mean_values = [(item * 255) for item in model_cfg["mean"]]
        scale_values = [(item * 255) for item in model_cfg["std"]]
        preprocess_fn = get_preprocess_fn(
            change_channel_order=True,
            input_size=model_cfg["input_size"][1:],
            mean=mean_values,
            std=scale_values,
            hwc_to_chw=True,
        )

        explain_method = None
        postprocess_fn = None
        if explain_mode == ExplainMode.WHITEBOX:
            if model_id in LIMITED_DIVERSE_SET_OF_CNN_MODELS:
                explain_method = Method.RECIPROCAM
            elif model_id in LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS:
                explain_method = Method.VITRECIPROCAM
            else:
                raise ValueError
        else:  # explain_mode == ExplainMode.BLACKBOX:
            postprocess_fn = get_postprocess_fn()

        explainer = Explainer(
            model=model_path,
            task=Task.CLASSIFICATION,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            explain_mode=explain_mode,
            explain_method=explain_method,
            embed_scaling=False,
        )

        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        explanation = explainer(
            image,
            targets=[target_class],
            resize=False,
            colormap=False,
            num_iterations_per_kernel=2,  # minimal iterations for feature test
        )

        assert explanation is not None
        assert explanation.shape[-1] > 1 and explanation.shape[-2] > 1
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.shape}.")
        self.clear_cache()

    @pytest.mark.parametrize(
        "model_id",
        [
            "resnet18.a1_in1k",
            "efficientnet_b0.ra_in1k",
            "vit_tiny_patch16_224.augreg_in21k",
            "deit_tiny_patch16_224.fb_in1k",
        ],
    )
    @pytest.mark.parametrize("detect", ["auto", "name"])
    def test_torch_insert_xai_with_layer(self, model_id: str, detect: str):
        xai_cfg = {
            "resnet18.a1_in1k": ("layer4", Method.RECIPROCAM),
            "efficientnet_b0.ra_in1k": ("bn2", Method.RECIPROCAM),
            "vit_tiny_patch16_224.augreg_in21k": ("blocks.9.norm1", Method.VITRECIPROCAM),
            "deit_tiny_patch16_224.fb_in1k": ("blocks.9.norm1", Method.VITRECIPROCAM),
        }

        model_dir = self.data_dir / "timm_models" / "converted_models"
        model, model_cfg = self.get_timm_model(model_id, model_dir)

        target_layer = xai_cfg[model_id][0] if detect == "name" else None
        explain_method = xai_cfg[model_id][1]

        image = cv2.imread("tests/assets/cheetah_person.jpg")
        image = cv2.resize(image, dsize=model_cfg["input_size"][1:])
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        mean = np.array(model.default_cfg["mean"])
        std = np.array(model.default_cfg["std"])
        image_norm = (image / 255.0 - mean) / std
        image_norm = image_norm.transpose((2, 0, 1))  # HWC -> CHW
        image_norm = image_norm[None, :]  # CHW -> 1CHW
        target_class = self.supported_num_classes[model_cfg["num_classes"]]

        model_xai: torch.nn.Module = insert_xai(
            model,
            task=Task.CLASSIFICATION,
            target_layer=target_layer,
            explain_method=explain_method,
        )

        with torch.no_grad():
            model_xai.eval()
            outputs = model_xai(torch.from_numpy(image_norm).float())
            logits = outputs["prediction"]
            probs = torch.softmax(logits, dim=-1)
            label = probs.argmax(dim=-1)[0]
        assert label.item() == target_class
        assert probs[0, label].item() > 0.0

        saliency_map: np.ndarray = outputs["saliency_map"].numpy(force=True)
        saliency_map = saliency_map.squeeze(0)
        assert saliency_map.shape[-1] > 1 and saliency_map.shape[-2] > 1
        assert saliency_map.min() < saliency_map.max()
        assert saliency_map.dtype == np.uint8

        self.clear_cache()

    @pytest.mark.parametrize(
        "explain_mode, explain_method",
        [
            (ExplainMode.WHITEBOX, Method.RECIPROCAM),
            (ExplainMode.WHITEBOX, Method.ACTIVATIONMAP),
            (ExplainMode.BLACKBOX, Method.AISE),
            (ExplainMode.BLACKBOX, Method.RISE),
        ],
    )
    def test_reference_map(self, explain_mode, explain_method):
        model_id = "resnet18.a1_in1k"
        model_dir = self.data_dir / "timm_models" / "converted_models"
        _, model_cfg = self.get_timm_model(model_id, model_dir)

        ir_path = model_dir / model_id / "model_fp32.xml"
        model = ov.Core().read_model(ir_path)

        mean_values = [(item * 255) for item in model_cfg["mean"]]
        scale_values = [(item * 255) for item in model_cfg["std"]]
        preprocess_fn = get_preprocess_fn(
            change_channel_order=True,
            input_size=model_cfg["input_size"][1:],
            mean=mean_values,
            std=scale_values,
            hwc_to_chw=True,
        )

        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=explain_mode,
            explain_method=explain_method,
            embed_scaling=False,
        )

        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        explanation = explainer(
            image,
            original_input_image=image,
            targets=[target_class],
            resize=False,
            colormap=False,
        )

        if explain_method == Method.ACTIVATIONMAP:
            generated_map = explanation.saliency_map["per_image_map"]
        else:
            generated_map = explanation.saliency_map[target_class]

        reference_maps_path = Path("tests/assets/reference_maps")
        reference_map = np.load(reference_maps_path / self.reference_maps_names[(explain_mode, explain_method)])
        assert np.all(np.abs(generated_map.astype(np.int16) - reference_map.astype(np.int16)) <= 3)

    def check_for_saved_map(self, model_id, directory):
        for target in self.supported_num_classes.values():
            map_name = model_id + "_target_" + str(target) + ".jpg"
            map_path = self.output_dir / directory / map_name
            map_saved = map_path.is_file()
            if map_saved:
                saved_map = cv2.imread(map_path._str)
                saved_map_shape = saved_map.shape
                shape = "H=" + str(saved_map_shape[0]) + ", W=" + str(saved_map_shape[1])
                self.update_report("report_wb.csv", model_id, "True", "True", "True", shape, str(map_saved))
                self.clear_cache()
                pytest.skip(f"Model {model_id} is already explained.")

    def get_timm_model(self, model_id: str, model_dir: Path):
        timm_model = timm.create_model(model_id, in_chans=3, pretrained=True, checkpoint_path="")
        timm_model.eval()
        model_cfg = timm_model.default_cfg
        num_classes = model_cfg["num_classes"]
        if num_classes not in self.supported_num_classes:
            self.clear_cache()
            pytest.skip(f"Number of model classes {num_classes} unknown")
        model_dir = model_dir / model_id
        ir_path = model_dir / "model_fp32.xml"
        if not ir_path.is_file():
            model_dir.mkdir(parents=True, exist_ok=True)
            ir_path = model_dir / "model_fp32.xml"
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = model_dir / "model_fp32.onnx"
            set_dynamic_batch = model_id in LIMITED_DIVERSE_SET_OF_VISION_TRANSFORMER_MODELS
            export_to_onnx(timm_model, onnx_path, dummy_tensor, set_dynamic_batch)
            export_to_ir(onnx_path, model_dir / "model_fp32.xml")
        return timm_model, model_cfg

    @staticmethod
    def put_confidence_into_map_overlay(explanation, target_confidence, target_class):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        if target_confidence > 0.5:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        thickness = 2
        map_ = cv2.putText(
            explanation.saliency_map[target_class],
            f"{target_confidence:.2f}",
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        explanation.saliency_map[target_class] = map_

    def update_report(
        self,
        report_name,
        model_id,
        exported_to_onnx="False",
        exported_to_ov_ir="False",
        explained="False",
        saliency_map_size="-",
        map_saved="False",
    ):
        fields = [model_id, exported_to_onnx, exported_to_ov_ir, explained, saliency_map_size, map_saved]
        last_row = self.report[-1]
        if last_row[0] != model_id:
            self.report.append(fields)
        else:
            for i in range(len(last_row)):
                if last_row[i] != fields[i]:
                    last_row[i] = fields[i]
            bool_flags = np.array(
                [[self.count(model[1]), self.count(model[2]), self.count(model[3])] for model in self.report[2:]]
            )
            self.report[1][1] = str(bool_flags[:, 0].sum())
            self.report[1][2] = str(bool_flags[:, 1].sum())
            self.report[1][3] = str(bool_flags[:, 2].sum())
        with open(self.output_dir / f"timm_{report_name}", "w") as f:
            write = csv.writer(f)
            write.writerows(self.report)

    def clear_cache(self):
        if self.clear_cache_converted_models:
            ir_model_dir = self.output_dir / "timm_models" / "converted_models"
            if ir_model_dir.is_dir():
                shutil.rmtree(ir_model_dir)
        if self.clear_cache_hf_models:
            cache_dir = os.environ.get("XDG_CACHE_HOME", "~/.cache")
            huggingface_hub_dir = Path(cache_dir).expanduser() / "huggingface/hub/"
            if huggingface_hub_dir.is_dir():
                shutil.rmtree(huggingface_hub_dir)

    @staticmethod
    def count(bool_string):
        if bool_string == "True":
            return 1
        if bool_string == "False":
            return 0
        raise ValueError


class TestExample:
    """Test sanity of examples/run_torch_onnx.py."""

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    def test_torch_onnx(self, tmp_path_factory: pytest.TempPathFactory):
        output_root = tmp_path_factory.mktemp("openvino_xai")
        output_dir = Path(output_root) / "example"
        cmd = [
            "python",
            "examples/run_torch_onnx.py",
            "--output_dir",
            output_dir,
        ]
        subprocess.run(cmd, check=True)  # noqa: S603, PLW1510
