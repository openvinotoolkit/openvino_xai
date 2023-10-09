import csv
import shutil

import cv2
import numpy as np
import pytest

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict

from openvino.model_api.models import ClassificationModel

from openvino_xai.explain import WhiteBoxExplainer, RISEExplainer
from openvino_xai.model import XAIClassificationModel
from openvino_xai.parameters import ClassificationExplainParametersWB, PostProcessParameters
from openvino_xai.saliency_map import TargetExplainGroup, PostProcessor

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")


class Command:
    def __init__(self, cmd: str, cwd: Path = None, env: Dict = None):
        self.cmd = cmd
        self.process = None
        self.exec_time = -1
        self.output = []  # store output here
        self.kwargs = {}
        self.timeout = False
        self.cwd = cwd
        self.env = env if env is not None else os.environ.copy()
        self.thread_exc = None

        self.kwargs.update(start_new_session=True)

    def kill_process_tree(self, pid):
        try:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
        except OSError as err:
            print(err)

    def run(self, timeout=3600, assert_returncode_zero=True):
        print(f"Running command: {self.cmd}")

        def target():
            try:
                start_time = time.time()
                with subprocess.Popen(
                    self.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    bufsize=1,
                    cwd=self.cwd,
                    env=self.env,
                    **self.kwargs,
                ) as p:
                    self.process = p
                    self.timeout = False

                    self.output = []
                    for line in self.process.stdout:
                        line = line.decode("utf-8")
                        self.output.append(line)
                        sys.stdout.write(line)

                    sys.stdout.flush()
                    self.process.stdout.close()

                    self.process.wait()
                    self.exec_time = time.time() - start_time
            except Exception as e:  # pylint:disable=broad-except
                self.thread_exc = e

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)

        if self.thread_exc is not None:
            raise self.thread_exc

        if thread.is_alive():
            try:
                print("Error: process taking too long to complete--terminating" + ", [ " + self.cmd + " ]")
                self.kill_process_tree(self.process.pid)
                self.exec_time = timeout
                self.timeout = True
                thread.join()
            except OSError as e:
                print(self.process.pid, "Exception when try to kill task by PID, " + e.strerror)
                raise
        returncode = self.process.wait()
        print("Process returncode = " + str(returncode))
        if assert_returncode_zero:
            assert returncode == 0, "Process exited with a non-zero exit code {}; output:{}".format(
                returncode, "".join(self.output)
            )
        return returncode

    def get_execution_time(self):
        return self.exec_time


def export_to_onnx(model: torch.nn.Module, save_path: str, data_sample: torch.Tensor) -> None:
    """
    Export Torch model to ONNX format.
    """
    torch.onnx.export(
        model, data_sample, save_path, export_params=True, opset_version=14, do_constant_folding=False
    )


def export_to_ir(model_path: str, save_path: str, model_name: str) -> None:
    """
    Export ONNX model to OpenVINO format.

    :param model_path: Path to ONNX model.
    :param save_path: Path directory to save OpenVINO IR model.
    :param model_name: Model name.
    """
    runner = Command(f"mo -m {model_path} -o {save_path} -n {model_name}")
    runner.run()


TEST_MODELS = timm.list_models(pretrained=True)


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


# TODO: include it into the report with corresponding error
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


LIMITED_DIVERSE_SET_OF_MODELS = [
    "resnet18.a1_in1k",
    "mobilenetv3_large_100.ra_in1k",
    "tf_efficientnet_b0.aa_in1k",
    "bat_resnext26ts.ch_in1k",
    "beit_base_patch16_224.in22k_ft_in22k",
    "botnet26t_256.c1_in1k",
    "caformer_b36.sail_in1k",
    "cait_m36_384.fb_dist_in1k",
    "coat_lite_mini.in1k",
    "coatnet_0_rw_224.sw_in1k",
    "convformer_b36.sail_in1k",
    "convit_tiny.fb_in1k",
    "convmixer_768_32.in1k",
    "convnext_base.clip_laion2b_augreg_ft_in1k",
    "convnextv2_pico.fcmae_ft_in1k",
    "crossvit_9_240.in1k",
    "cs3darknet_l.c2ns_in1k",
    "darknet53.c2ns_in1k",
    "davit_tiny.msft_in1k",
    "deit_tiny_patch16_224.fb_in1k",
    "densenet121.ra_in1k",
    "dla34.in1k",
    "dpn68.mx_in1k",
    "eca_botnext26ts_256.c1_in1k",
    "ecaresnet26t.ra2_in1k",
    "edgenext_base.in21k_ft_in1k",
    "efficientformer_l1.snap_dist_in1k",
    "efficientnet_b0.ra_in1k",
    "ese_vovnet19b_dw.ra_in1k",
    "eva02_base_patch14_224.mim_in22k",
    "fbnetv3_b.ra2_in1k",
    "flexivit_small.300ep_in1k",
    "focalnet_base_lrf.ms_in1k",
    "gcresnet33ts.ra2_in1k",
    "gcvit_tiny.in1k",
    "gernet_s.idstcv_in1k",
    "hardcorenas_a.miil_green_in1k",
    "hrnet_w18.ms_aug_in1k",
    "inception_v3.gluon_in1k",
    "lambda_resnet26rpt_256.c1_in1k",
    "lcnet_050.ra2_in1k",
    "legacy_senet154.in1k",
    "levit_128.fb_dist_in1k",
    "levit_conv_128.fb_dist_in1k",
    "maxvit_base_tf_224.in1k",
    "mixer_b16_224.goog_in21k",
    "mixnet_s.ft_in1k",
    "mobilenetv2_100.ra_in1k",
    "mobilevitv2_050.cvnets_in1k",
    "mvitv2_base.fb_in1k",
    "nest_tiny_jx.goog_in1k",
    "pit_s_224.in1k",
    "poolformer_m36.sail_in1k",
    "pvt_v2_b0.in1k",
    "regnety_002.pycls_in1k",
    "repvgg_a2.rvgg_in1k",
    "repvit_m1.dist_in1k",
    "res2net50_14w_8s.in1k",
    "resmlp_12_224.fb_dino",
    "resnest14d.gluon_in1k",
    "resnetaa50.a1h_in1k",
    "resnetrs50.tf_in1k",
    "resnext26ts.ra2_in1k",
    "rexnet_100.nav_in1k",
    "samvit_base_patch16.sa1b",
    "selecsls42b.in1k",
    "sequencer2d_l.in1k",
    "seresnet50.a1_in1k",
    "seresnext26d_32x4d.bt_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "swinv2_tiny_window8_256.ms_in1k",
    "tf_mixnet_l.in1k",
    "tf_mobilenetv3_large_075.in1k",
    "tinynet_a.in1k",
    "tresnet_l.miil_in1k",
    "twins_svt_small.in1k",
    "vgg11.tv_in1k",
    "vit_tiny_patch16_224.augreg_in21k",
    "wide_resnet50_2.racm_in1k",
    "xception41.tf_in1k",
    "xcit_nano_12_p8_224.fb_dist_in1k",
]


class TestImageClassificationTimm:
    data_dir = Path(".data")
    fields = ["Model", "Exported to ONNX", "Exported to OV IR", "Explained", "Map size", "Map saved"]
    counter_row = ["Counters", "0", "0", "0", "-", "-"]
    report = [fields, counter_row]
    clean_cash_converted_models = True
    clean_cash_hf_models = True
    supported_num_classes = {
        1000: 293,  # 293 is a cheetah class_id in the ImageNet-1k dataset
        21841: 2441,  # 2441 is a cheetah class_id in the ImageNet-21k dataset
        11821: 1652,  # 1652 is a cheetah class_id in the ImageNet-12k dataset
    }

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_cnn_classification_white_box(
            self, model_id, dump_maps=True
    ):
        if not any(cnn_model_key in model_id for cnn_model_key in CNN_MODELS):
            pytest.skip(f"Model {model_id} is not CNN-based.")
        if any(cnn_model_key in model_id for cnn_model_key in NON_CONVERTABLE_CNN_MODELS):
            pytest.skip(f"Model {model_id} is non-convertable ether to ONNX or to OV representation.")

        self.check_for_saved_map(model_id, "timm_models/maps_wb/")

        output_model_dir = self.data_dir / "timm_models" / "converted_models" / model_id
        output_model_dir.mkdir(parents=True, exist_ok=True)
        ir_path = output_model_dir / "model_fp32.xml"

        timm_model, model_cfg = self.get_timm_model(model_id)

        self.update_report("report_wb.csv", model_id)
        if not (output_model_dir / "model_fp32.xml").is_file():
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            export_to_onnx(timm_model, onnx_path, dummy_tensor)
            self.update_report("report_wb.csv", model_id, "True")
            export_to_ir(onnx_path, output_model_dir, model_name="model_fp32")
            self.update_report("report_wb.csv", model_id, "True", "True")
        else:
            self.update_report("report_wb.csv", model_id, "True", "True")

        mapi_params = self.get_mapi_params(model_cfg)
        model = XAIClassificationModel.create_model(
            ir_path,
            "Classification",
            **mapi_params,
            explain_parameters=ClassificationExplainParametersWB(embed_normalization=False),
        )

        image = cv2.imread("tests/assets/cheetah_person.jpg")
        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        explanation = WhiteBoxExplainer(model).explain(
            image,
            TargetExplainGroup.CUSTOM_CLASSES,
            [target_class],
        )

        assert explanation is not None
        assert explanation.sal_map_shape[-1] > 1 and explanation.sal_map_shape[-2] > 1
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.sal_map_shape}.")
        self.update_report("report_wb.csv", model_id, "True", "True", "True")
        raw_shape = explanation.sal_map_shape
        shape_str = "H=" + str(raw_shape[0]) + ", W=" + str(raw_shape[1])
        self.update_report("report_wb.csv", model_id, "True", "True", "True", shape_str)

        if dump_maps:
            # timm workaround to remove outlier activations at corners
            raw_sal_map = explanation.map[target_class]
            raw_sal_map[0, 0] = np.mean(np.delete(raw_sal_map[:2, :2].flatten(), 0))
            raw_sal_map[0, -1] = np.mean(np.delete(raw_sal_map[:2, -2:].flatten(), 1))
            raw_sal_map[-1, 0] = np.mean(np.delete(raw_sal_map[-2:, :2].flatten(), 2))
            raw_sal_map[-1, -1] = np.mean(np.delete(raw_sal_map[-2:, -2:].flatten(), 3))
            explanation.map[target_class] = raw_sal_map
            post_processing_parameters = PostProcessParameters(normalize=True, overlay=True)
            post_processor = PostProcessor(
                explanation,
                image,
                post_processing_parameters,
            )
            explanation = post_processor.postprocess()

            target_confidence = model(image).raw_scores[target_class]
            self.put_confidence_into_map_overlay(explanation, target_confidence, target_class)

            save_dir = self.data_dir / "timm_models" / "maps_wb"
            explanation.save(save_dir, model_id)
            file_name = model_id + "_target_" + str(target_class) + ".jpg"
            map_saved = (save_dir / file_name).is_file()
            self.update_report("report_wb.csv", model_id, "True", "True", "True", shape_str, str(map_saved))
        self.clean_cash()

    # sudo ln -s /usr/local/cuda-11.8/ cuda
    # pip uninstall torch torchvision
    # pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
    #
    # ulimit -a
    # ulimit -Sn 10000
    # ulimit -a
    @pytest.mark.parametrize("model_id", LIMITED_DIVERSE_SET_OF_MODELS)
    def test_cnn_classification_black_box(
            self, model_id, dump_maps=True
    ):
        self.check_for_saved_map(model_id, "timm_models/maps_bb/")

        timm_model, model_cfg = self.get_timm_model(model_id)

        self.update_report("report_bb.csv", model_id)

        output_model_dir = self.data_dir / "timm_models" / "converted_models" / model_id
        output_model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_model_dir / "model_fp32.onnx"

        if not (output_model_dir / "model_fp32.onnx").is_file():
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            export_to_onnx(timm_model, onnx_path, dummy_tensor)
            self.update_report("report_bb.csv", model_id, "True", "True")
        else:
            self.update_report("report_bb.csv", model_id, "True", "True")

        mapi_params = self.get_mapi_params(model_cfg)
        model = ClassificationModel.create_model(
            onnx_path,
            model_type="Classification",
            **mapi_params,
        )

        explainer = RISEExplainer(model)
        image = cv2.imread("tests/assets/cheetah_person.jpg")
        target_class = self.supported_num_classes[model_cfg["num_classes"]]
        explanation = explainer.explain(
            image,
            TargetExplainGroup.CUSTOM_CLASSES,
            [target_class],
            post_processing_parameters=PostProcessParameters(overlay=True),
        )

        assert explanation is not None
        assert explanation.sal_map_shape[-1] > 1 and explanation.sal_map_shape[-2] > 1
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.sal_map_shape}.")
        self.update_report("report_bb.csv", model_id, "True", "True", "True")
        shape = explanation.sal_map_shape
        shape_str = "H=" + str(shape[0]) + ", W=" + str(shape[1])
        self.update_report("report_bb.csv", model_id, "True", "True", "True", shape_str)

        if dump_maps:
            target_confidence = model(image).raw_scores[target_class]
            self.put_confidence_into_map_overlay(explanation, target_confidence, target_class)

            save_dir = self.data_dir / "timm_models" / "maps_bb"
            explanation.save(save_dir, model_id)
            file_name = model_id + "_target_" + str(target_class) + ".jpg"
            map_saved = (save_dir / file_name).is_file()
            self.update_report("report_bb.csv", model_id, "True", "True", "True", shape_str, str(map_saved))
        self.clean_cash()

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
                self.clean_cash()
                pytest.skip(f"Model {model_id} is already explained.")

    def get_timm_model(self, model_id):
        timm_model = timm.create_model(
            model_id, in_chans=3, pretrained=True, checkpoint_path=""
        )
        model_cfg = timm_model.default_cfg
        num_classes = model_cfg["num_classes"]
        if num_classes not in self.supported_num_classes:
            self.clean_cash()
            pytest.skip(f"Number of model classes {num_classes} unknown")
        return timm_model, model_cfg

    @staticmethod
    def get_mapi_params(model_cfg):
        mapi_params = {
            "configuration": {
                "mean_values": [(item * 255) for item in model_cfg["mean"]],
                "scale_values": [(item * 255) for item in model_cfg["std"]],
                "output_raw_scores": True,
            }
        }
        return mapi_params

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
            explanation.map[target_class], f"{target_confidence:.2f}", org, font, fontScale, color, thickness, cv2.LINE_AA
        )
        explanation.map[target_class] = map_

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
                [
                    [self.count(model[1]), self.count(model[2]), self.count(model[3])]
                    for model in self.report[2:]
                ]
            )
            self.report[1][1] = str(bool_flags[:, 0].sum())
            self.report[1][2] = str(bool_flags[:, 1].sum())
            self.report[1][3] = str(bool_flags[:, 2].sum())
        with open(self.data_dir / "timm_models" / report_name, "w") as f:
            write = csv.writer(f)
            write.writerows(self.report)

    def clean_cash(self):
        if self.clean_cash_converted_models:
            ir_model_dir = self.data_dir / "timm_models" / "converted_models"
            if ir_model_dir.is_dir():
                shutil.rmtree(ir_model_dir)
        if self.clean_cash_hf_models:
            huggingface_hub_dir = Path(os.path.expanduser("~")) / ".cache/huggingface/hub/"
            if huggingface_hub_dir.is_dir():
                shutil.rmtree(huggingface_hub_dir)

    @staticmethod
    def count(bool_string):
        if bool_string == "True":
            return 1
        if bool_string == "False":
            return 0
        raise ValueError
