# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino

from openvino_xai.utils.torch import torch


def export_to_onnx(model: torch.nn.Module, save_path: str, data_sample: torch.Tensor, set_dynamic_batch: bool) -> None:  # type: ignore
    """
    Export Torch model to ONNX format.
    """
    dynamic_axes = {"data": {0: "batch"}} if set_dynamic_batch else dict()
    torch.onnx.export(
        model,
        data_sample,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=["data"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )


def export_to_ir(model_path: str, save_path: str) -> None:
    """
    Export ONNX model to OpenVINO format.

    :param model_path: Path to ONNX model.
    :param save_path: Filepath to save OpenVINO IR model.
    """
    ov_model = openvino.convert_model(model_path)
    openvino.save_model(ov_model, save_path)
