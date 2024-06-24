# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
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


TEST_MODELS = timm.list_models(pretrained=True)

CNN_MODELS = [
    "bat_resnext",
    "convnext",
    "cs3",
    "cs3darknet",
    "darknet",
    "densenet",
    "dla",
    "dpn",
    "efficientnet",
    "ese_vovnet",
    "fbnet",
    "gernet",
    "ghostnet",
    "hardcorenas",
    "hrnet",
    "inception",
    "lcnet",
    "legacy_",
    "mixnet",
    "mnasnet",
    "mobilenet",
    "nasnet",
    "regnet",
    "repvgg",
    "res2net",
    "res2next",
    "resnest",
    "resnet",
    "resnext",
    "rexnet",
    "selecsls",
    "semnasnet",
    "senet",
    "seresnext",
    "spnasnet",
    "tinynet",
    "tresnet",
    "vgg",
    "xception",
]

SUPPORTED_BUT_FAILED_BY_BB_MODELS = {}

NOT_SUPPORTED_BY_BB_MODELS = {
    "_nfnet_": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "convnext_xxlarge": "RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.",
    "convnextv2_huge": "RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.",
    "deit3_huge": "RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.",
    "dm_nfnet": "openvino._pyopenvino.GeneralFailure: Check 'false' failed at src/frontends/onnx/frontend/src/frontend.cpp:144",
    "eca_nfnet": "openvino._pyopenvino.GeneralFailure: Check 'false' failed at src/frontends/onnx/frontend/src/frontend.cpp:144",
    "eva_giant": "RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.",
    "halo": "torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of operator Unfold, input size not accessible.",
    "nf_regnet": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "nf_resnet": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "nfnet_l0": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "regnety_1280": "RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.",
    "regnety_2560": "RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.",
    "repvit": "urllib.error.HTTPError: HTTP Error 404: Not Found",
    "resnetv2": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "tf_efficientnet_cc": "torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of convolution for kernel of unknown shape.",
    "vit_base_r50_s16_224.orig_in21k": "RuntimeError: Error(s) in loading state_dict for VisionTransformer",
    "vit_gigantic_patch16_224_ijepa.in22k": "RuntimeError: shape '[1, 13, 13, -1]' is invalid for input of size 274560",
    "vit_huge_patch14_224.orig_in21k": "RuntimeError: Error(s) in loading state_dict for VisionTransformer",
    "vit_large_patch32_224.orig_in21k": "RuntimeError: Error(s) in loading state_dict for VisionTransformer",
    "vit_large_r50_s32": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "vit_small_r26_s32": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "vit_tiny_r_s16": "RuntimeError: Exception from src/inference/src/cpp/core.cpp:90: Training mode of BatchNormalization is not supported.",
    "volo_": "torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::col2im' to ONNX opset version 14 is not supported.",
}

SUPPORTED_BUT_FAILED_BY_WB_MODELS = {
    "convformer": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "swin": "Only two outputs of the between block Add node supported, but got 1. Try to use black-box.",
}

NOT_SUPPORTED_BY_WB_MODELS = {
    **NOT_SUPPORTED_BY_BB_MODELS,
    # Killed on WB
    "beit_large_patch16_512": "Failed to allocate 94652825600 bytes of memory",
    "eva02_base_patch14_448": "OOM Killed",
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
    # Not expected to work for now
    "botnet26t_256": "Only two outputs of the between block Add node supported, but got 1",
    "caformer": "One (and only one) of the nodes has to be Add type. But got MVN and Multiply.",
    "cait_": "Cannot create an empty Constant. Please provide valid data.",
    "coat_": "Only two outputs of the between block Add node supported, but got 1.",
    "coatn": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "convmixer": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "crossvit": "One (and only one) of the nodes has to be Add type. But got StridedSlice and StridedSlice.",
    "davit": "Only two outputs of the between block Add node supported, but got 1.",
    "edgenext": "Only two outputs of the between block Add node supported, but got 1",
    "efficientformer": "Cannot find output backbone_node in auto mode.",
    "focalnet": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "gcvit": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "levit_": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "maxvit": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "maxxvit": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "mobilevitv2": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "nest_": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "poolformer": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "sebotnet": "Only two outputs of the between block Add node supported, but got 1.",
    "sequencer2d": "Cannot find output backbone_node in auto mode, please provide target_layer.",
    "tnt_s_patch16_224": "Only two outputs of the between block Add node supported, but got 1.",
    "tresnet": "Batch shape of the output should be dynamic, but it is static.",
    "twins": "One (and only one) of the nodes has to be Add type. But got ShapeOf and Transpose.",
    "visformer": "Cannot find output backbone_node in auto mode, please provide target_layer",
    "vit_relpos_base_patch32_plus_rpn_256": "Check 'TRShape::merge_into(output_shape, in_copy)' failed",
    "vit_relpos_medium_patch16_rpn_224": "ValueError in openvino_xai/methods/white_box/recipro_cam.py:215",
}


class TestImageClassificationTimm:
    fields = ["Model", "Exported to ONNX", "Exported to OV IR", "Explained", "Map size", "Map saved"]
    counter_row = ["Counters", "0", "0", "0", "-", "-"]
    report = [fields, counter_row]
    clean_cache_converted_models = False
    clean_cache_hf_models = False
    supported_num_classes = {
        1000: 293,  # 293 is a cheetah class_id in the ImageNet-1k dataset
        21841: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        21843: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        11821: 1652,  # 1652 is a cheetah class_id in the ImageNet-12k dataset
    }

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root, fxt_output_root):
        self.data_dir = fxt_data_root
        self.output_dir = fxt_output_root

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_white_box(self, model_id, dump_maps=False):
        # self.check_for_saved_map(model_id, "timm_models/maps_wb/")

        for skipped_model in NOT_SUPPORTED_BY_WB_MODELS.keys():
            if skipped_model in model_id:
                pytest.skip(reason=NOT_SUPPORTED_BY_WB_MODELS[skipped_model])

        for failed_model in SUPPORTED_BUT_FAILED_BY_WB_MODELS.keys():
            if failed_model in model_id:
                pytest.xfail(reason=SUPPORTED_BUT_FAILED_BY_WB_MODELS[failed_model])

        explain_method = Method.VITRECIPROCAM
        for cnn_model in CNN_MODELS:
            if cnn_model in model_id:
                explain_method = Method.RECIPROCAM
                break

        timm_model, model_cfg = self.get_timm_model(model_id)
        self.update_report("report_wb.csv", model_id)

        ir_path = self.data_dir / "timm_models" / "converted_models" / model_id / "model_fp32.xml"
        if not ir_path.is_file():
            output_model_dir = self.output_dir / "timm_models" / "converted_models" / model_id
            output_model_dir.mkdir(parents=True, exist_ok=True)
            ir_path = output_model_dir / "model_fp32.xml"
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            set_dynamic_batch = explain_method == Method.VITRECIPROCAM
            export_to_onnx(timm_model, onnx_path, dummy_tensor, set_dynamic_batch)
            self.update_report("report_wb.csv", model_id, "True")
            export_to_ir(onnx_path, output_model_dir / "model_fp32.xml")
            self.update_report("report_wb.csv", model_id, "True", "True")
        else:
            self.update_report("report_wb.csv", model_id, "True", "True")

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
        self.clean_cache()

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_black_box(self, model_id, dump_maps=False):
        # self.check_for_saved_map(model_id, "timm_models/maps_bb/")

        for skipped_model in NOT_SUPPORTED_BY_BB_MODELS.keys():
            if skipped_model in model_id:
                pytest.skip(reason=NOT_SUPPORTED_BY_BB_MODELS[skipped_model])

        for failed_model in SUPPORTED_BUT_FAILED_BY_BB_MODELS.keys():
            if failed_model in model_id:
                pytest.xfail(reason=SUPPORTED_BUT_FAILED_BY_BB_MODELS[failed_model])

        timm_model, model_cfg = self.get_timm_model(model_id)
        self.update_report("report_bb.csv", model_id)

        onnx_path = self.data_dir / "timm_models" / "converted_models" / model_id / "model_fp32.onnx"
        if not onnx_path.is_file():
            output_model_dir = self.output_dir / "timm_models" / "converted_models" / model_id
            output_model_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = output_model_dir / "model_fp32.onnx"
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            export_to_onnx(timm_model, onnx_path, dummy_tensor, False)
            self.update_report("report_bb.csv", model_id, "True", "True")
        else:
            self.update_report("report_bb.csv", model_id, "True", "True")

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
            # num_masks=2000,  # kwargs of the RISE algo
            num_masks=2,  # minimal iterations for feature test
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
        self.clean_cache()

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
                self.clean_cache()
                pytest.skip(f"Model {model_id} is already explained.")

    def get_timm_model(self, model_id):
        timm_model = timm.create_model(model_id, in_chans=3, pretrained=True, checkpoint_path="")
        timm_model.eval()
        model_cfg = timm_model.default_cfg
        num_classes = model_cfg["num_classes"]
        if num_classes not in self.supported_num_classes:
            self.clean_cache()
            pytest.skip(f"Number of model classes {num_classes} unknown")
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

    def clean_cache(self):
        if self.clean_cache_converted_models:
            ir_model_dir = self.output_dir / "timm_models" / "converted_models"
            if ir_model_dir.is_dir():
                shutil.rmtree(ir_model_dir)
        if self.clean_cache_hf_models:
            cache_dir = os.environ.get("XDG_CACHE_HOME", "~/.cache")
            huggingface_hub_dir = Path(cache_dir) / "huggingface/hub/"
            if huggingface_hub_dir.is_dir():
                shutil.rmtree(huggingface_hub_dir)

    @staticmethod
    def count(bool_string):
        if bool_string == "True":
            return 1
        if bool_string == "False":
            return 0
        raise ValueError
