# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
import pytest

from openvino_xai.common.parameters import Method, Task
from openvino_xai.explainer.explainer import Explainer
from openvino_xai.explainer.parameters import (
    ExplainMode,
    ExplanationParameters,
    TargetExplainGroup,
    VisualizationParameters,
)
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
    get_score,
)
from openvino_xai.explainer.visualizer import Visualizer
from openvino_xai.inserter.parameters import ClassificationInsertionParameters
from openvino_xai.utils.model_export import export_to_ir, export_to_onnx

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")


TEST_MODELS = timm.list_models(pretrained=True)

NON_SUPPORTED_BY_WB_MODELS = {
    "repvit_m1.dist_in1k":  "404 Not Found",
    "nest_tiny_jx.goog_in1k": "CNN, dynamic batch issue",
    "pit_s_224.in1k": "CNN, dynamic batch issue",
    "pvt_v2_b0.in1k": "CNN, dynamic batch issue",
    "sequencer2d_l.in1k": "CNN, dynamic batch issue",
    "mobilevitv2_050.cvnets_in1k": "CNN, dynamic batch issue",
    "convformer_b36.sail_in1k": "Transformer, various issues",
    "davit_tiny.msft_in1k": "Transformer, various issues",
    "poolformer_m36.sail_in1k": "Transformer, various issues",
    "caformer_b36.sail_in1k": "Transformer, various issues",
    "cait_m36_384.fb_dist_in1k": "Transformer, various issues",
    "coat_lite_mini.in1k": "Transformer, various issues",
    "crossvit_9_240.in1k": "Transformer, various issues",
    "swin_tiny_patch4_window7_224.ms_in1k": "Transformer, various issues",
    "swinv2_tiny_window8_256.ms_in1k": "Transformer, various issues",
    "twins_svt_small.in1k": "Transformer, various issues",
    "efficientformer_l1.snap_dist_in1k": "Transformer, various issues",
    "gcvit_tiny.in1k": "Transformer, various issues",
    "levit_128.fb_dist_in1k": "Transformer, various issues",
    "maxvit_base_tf_224.in1k": "Transformer, various issues",
    "mvitv2_base.fb_in1k": "Transformer, various issues",
    "poolformer_m36.sail_in1k": "Transformer, various issues",
    "xcit_nano_12_p8_224.fb_dist_in1k": "Transformer, various issues",
    "convmixer_768_32.in1k": "Transformer, various issues",
}

NON_SUPPORTED_BY_BB_MODELS = {
    "repvit_m1.dist_in1k":  "404 Not Found",
}

CNN_MODELS = [
    "bat_resnext",
    "convnext",
    "cs3darknet",
    "cs3",
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
    "resnext",
    "rexnet",
    "selecsls",
    "semnasnet",
    "senet",
    "seresnext",
    "spnasnet",
    "tinynet",
    "vgg",
    "xception",
    "resnet",
]

NON_CONVERTABLE_CNN_MODELS = [
    "convnext_xxlarge",  # too big
    "convnextv2_huge",  # too big
    "gc_efficientnetv2_rw",  # failed to convert to OV
    "gcresnext",  # failed to convert to OV
    "haloregnetz",
    "nasnetalarge",
    "pnasnet5large",
    "regnety_1280",
    "regnety_2560",
    "resnest14d",
    "resnest26d",
    "resnest50d",
    "resnest101e",
    "resnest200e",
    "resnest269e",
    "skresnext50_32x4d",
    "tf_efficientnet_cc_b",
    "gcresnet",
    "lambda_resnet",
    "nf_regnet",
    "nf_resnet",
    "resnetv2_50x",
    "resnetv2_101x",
    "resnetv2_152x",
    "skresnet",
    "tresnet_",
]


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
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_white_box(self, model_id, dump_maps=False):
        # self.check_for_saved_map(model_id, "timm_models/maps_wb/")

        for non_supported_model in NON_SUPPORTED_BY_WB_MODELS.keys():
            if model_id in non_supported_model:
                pytest.xfail(reason=NON_SUPPORTED_BY_WB_MODELS[non_supported_model])

        output_model_dir = self.data_dir / "timm_models" / "converted_models" / model_id
        output_model_dir.mkdir(parents=True, exist_ok=True)
        ir_path = output_model_dir / "model_fp32.xml"

        timm_model, model_cfg = self.get_timm_model(model_id)
        self.update_report("report_wb.csv", model_id)

        explain_method = Method.VITRECIPROCAM
        for cnn_model in CNN_MODELS:
            if cnn_model in model_id:
                explain_method = Method.RECIPROCAM
                break

        if not (output_model_dir / "model_fp32.xml").is_file():
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            set_dynamic_batch = (explain_method == Method.VITRECIPROCAM)
            export_to_onnx(timm_model, onnx_path, dummy_tensor, set_dynamic_batch)
            self.update_report("report_wb.csv", model_id, "True")
            export_to_ir(onnx_path, output_model_dir / "model_fp32.xml")
            self.update_report("report_wb.csv", model_id, "True", "True")
        else:
            self.update_report("report_wb.csv", model_id, "True", "True")

        model = ov.Core().read_model(ir_path)

        insertion_parameters = ClassificationInsertionParameters(
            embed_scaling=False,
            explain_method=explain_method,
        )

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
            insertion_parameters=insertion_parameters,
        )

        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.CUSTOM,
            target_explain_labels=[target_class],
            visualization_parameters=VisualizationParameters(),
        )
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        explanation = explainer(image, explanation_parameters)

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
            visualization_parameters = VisualizationParameters(scaling=True, overlay=True)
            post_processor = Visualizer(
                explanation=explanation,
                original_input_image=image,
                visualization_parameters=visualization_parameters,
            )
            explanation = post_processor.run()

            model_output = explainer.model_forward(image)
            target_confidence = get_score(model_output["logits"], target_class, activation=ActivationType.SOFTMAX)
            self.put_confidence_into_map_overlay(explanation, target_confidence, target_class)

            save_dir = self.data_dir / "timm_models" / "maps_wb"
            explanation.save(save_dir, model_id)
            file_name = model_id + "_target_" + str(target_class) + ".jpg"
            map_saved = (save_dir / file_name).is_file()
            self.update_report("report_wb.csv", model_id, "True", "True", "True", shape_str, str(map_saved))
        self.clean_cache()

    # sudo ln -s /usr/local/cuda-11.8/ cuda
    # pip uninstall torch torchvision
    # pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
    #
    # ulimit -a
    # ulimit -Sn 10000
    # ulimit -a
    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_classification_black_box(self, model_id, dump_maps=False):
        # self.check_for_saved_map(model_id, "timm_models/maps_bb/")

        for non_supported_model in NON_SUPPORTED_BY_BB_MODELS.keys():
            if model_id in non_supported_model:
                pytest.xfail(reason=NON_SUPPORTED_BY_BB_MODELS[non_supported_model])

        timm_model, model_cfg = self.get_timm_model(model_id)

        self.update_report("report_bb.csv", model_id)

        output_model_dir = self.data_dir / "timm_models" / "converted_models" / model_id
        output_model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_model_dir / "model_fp32.onnx"

        if not (output_model_dir / "model_fp32.onnx").is_file():
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
        explanation_parameters = ExplanationParameters(
            target_explain_group=TargetExplainGroup.CUSTOM,
            target_explain_labels=[target_class],
        )
        explanation = explainer(
            image,
            explanation_parameters=explanation_parameters,
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

            save_dir = self.data_dir / "timm_models" / "maps_bb"
            explanation.save(save_dir, model_id)
            file_name = model_id + "_target_" + str(target_class) + ".jpg"
            map_saved = (save_dir / file_name).is_file()
            self.update_report("report_bb.csv", model_id, "True", "True", "True", shape_str, str(map_saved))
        self.clean_cache()

    def check_for_saved_map(self, model_id, directory):
        for target in self.supported_num_classes.values():
            map_name = model_id + "_target_" + str(target) + ".jpg"
            map_path = self.data_dir / directory / map_name
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
        with open(self.data_dir / "timm_models" / report_name, "w") as f:
            write = csv.writer(f)
            write.writerows(self.report)

    def clean_cache(self):
        if self.clean_cache_converted_models:
            ir_model_dir = self.data_dir / "timm_models" / "converted_models"
            if ir_model_dir.is_dir():
                shutil.rmtree(ir_model_dir)
        if self.clean_cache_hf_models:
            huggingface_hub_dir = Path.home() / ".cache/huggingface/hub/"
            if huggingface_hub_dir.is_dir():
                shutil.rmtree(huggingface_hub_dir)

    @staticmethod
    def count(bool_string):
        if bool_string == "True":
            return 1
        if bool_string == "False":
            return 0
        raise ValueError
