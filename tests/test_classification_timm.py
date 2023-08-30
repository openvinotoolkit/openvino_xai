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

from openvino_xai.explain import WhiteBoxExplainer
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
    torch.onnx.export(model, data_sample, save_path, export_params=True, opset_version=13, do_constant_folding=False)


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
    "cspresnet",
    "densenet",
    "dla",
    "dpn",
    "eca_resnet",
    "ecaresnet",
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
    "resnest",
    "resnext",
    "rexnet",
    "selecsls",
    "semnasnet",
    "senet",
    "seresnet",
    "seresnext",
    "spnasnet",
    "tinynet",
    "vgg",
    "xception",
]


NON_CONVERTABLE_CNN_MODELS = [
    "convnext_xxlarge",  # too big
    "convnextv2_huge",  # too big
]


class TestImageClassificationTimm:
    data_dir = ".data"
    fields = ["Model", "Exported to ONNX", "Exported to OV IR", "Explained", "Map size", "Map saved"]
    counter_row = ["Counters", "0", "0", "0", "-", "-"]
    report = [fields, counter_row]

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_cnn_classification_white_box(
            self, model_id, clean_ir_cash=True, clean_huggingface_cash=True, dump_maps=True
    ):
        self.clean_cash(clean_ir_cash, clean_huggingface_cash)
        if not any(cnn_model_key in model_id for cnn_model_key in CNN_MODELS):
            pytest.skip(f"Model {model_id} is not CNN-based.")
        if any(cnn_model_key in model_id for cnn_model_key in NON_CONVERTABLE_CNN_MODELS):
            pytest.skip(f"Model {model_id} is non-convertable ether to ONNX or to OV representation.")

        output_dir = os.path.join(self.data_dir, "timm_models", "converted_models", model_id)
        output_model_dir = Path(output_dir)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        ir_path = output_model_dir / "model_fp32.xml"

        self.update_report(model_id)

        timm_model = timm.create_model(model_id, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path="")
        if not os.path.isfile(os.path.join(output_model_dir, "model_fp32.xml")):
            input_size = [1] + list(timm_model.default_cfg["input_size"])
            dummy_tensor = torch.rand(input_size)
            onnx_path = output_model_dir / "model_fp32.onnx"
            export_to_onnx(timm_model, onnx_path, dummy_tensor)
            self.update_report(model_id, "True")
            export_to_ir(onnx_path, output_model_dir, model_name="model_fp32")
            self.update_report(model_id, "True", "True")
        else:
            self.update_report(model_id, "True", "True")

        model_cfg = timm_model.default_cfg
        mapi_params = {
            "configuration": {
                "mean_values": [(item * 255) for item in model_cfg["mean"]],
                "scale_values": [(item * 255) for item in model_cfg["std"]],
            }
        }
        explain_parameters = ClassificationExplainParametersWB(embed_normalization=False)
        model = XAIClassificationModel.create_model(
            ir_path, "Classification",
            **mapi_params,
            explain_parameters=explain_parameters
        )

        image = cv2.imread("assets/cheetah_class293.jpg")
        target_class = 293  # 293 is a cheetah class_id in the ImageNet dataset
        explanation = WhiteBoxExplainer(model).explain(
            image,
            TargetExplainGroup.CUSTOM_CLASSES,
            [target_class],
        )
        assert explanation is not None
        print(f"{model_id}: Generated classification saliency maps with shape {explanation.map.shape}.")
        self.update_report(model_id, "True", "True", "True")
        raw_shape = explanation.map.shape
        shape = "H=" + str(raw_shape[1]) + ", W=" + str(raw_shape[2])
        self.update_report(model_id, "True", "True", "True", shape)

        if dump_maps:
            # timm workaround to remove outlier activations at corners
            raw_sal_map = explanation.map[0]
            raw_sal_map[0, 0] = np.mean(np.delete(raw_sal_map[:2, :2].flatten(), 0))
            raw_sal_map[0, -1] = np.mean(np.delete(raw_sal_map[:2, -2:].flatten(), 1))
            raw_sal_map[-1, 0] = np.mean(np.delete(raw_sal_map[-2:, :2].flatten(), 2))
            raw_sal_map[-1, -1] = np.mean(np.delete(raw_sal_map[-2:, -2:].flatten(), 3))
            explanation.map = raw_sal_map[None, ...]

            post_processing_parameters = PostProcessParameters(normalize=True, overlay=True)
            post_processor = PostProcessor(
                explanation,
                image,
                post_processing_parameters.normalize,
                post_processing_parameters.resize,
                post_processing_parameters.colormap,
                post_processing_parameters.overlay,
                post_processing_parameters.overlay_weight,
            )

            explain_result = post_processor.postprocess()
            explain_result.save(os.path.join(self.data_dir, "timm_models/maps/"), model_id)
            map_saved = os.path.exists(
                os.path.join(os.path.join(self.data_dir, "timm_models/maps/"), model_id + ".jpg")
            )
            self.update_report(model_id, "True", "True", "True", shape, str(map_saved))
        self.clean_cash(clean_ir_cash, clean_huggingface_cash)

    def update_report(
            self,
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
            bool_flags = np.array([[bool(model[1]), bool(model[2]), bool(model[3])] for model in self.report[2:]])
            self.report[1][1] = str(bool_flags[:, 0].sum())
            self.report[1][2] = str(bool_flags[:, 1].sum())
            self.report[1][3] = str(bool_flags[:, 2].sum())
        with open(os.path.join(self.data_dir, "timm_models/report.csv"), "w") as f:
            write = csv.writer(f)
            write.writerows(self.report)

    def clean_cash(self, clean_ir_cash, clean_huggingface_cash):
        if clean_ir_cash:
            ir_model_dir = os.path.join(self.data_dir, "timm_models", "converted_models")
            if os.path.isdir(ir_model_dir):
                shutil.rmtree(ir_model_dir)
        if clean_huggingface_cash:
            huggingface_hub_dir = os.path.join(os.path.expanduser('~'), ".cache/huggingface/hub/")
            if os.path.isdir(huggingface_hub_dir):
                shutil.rmtree(huggingface_hub_dir)
