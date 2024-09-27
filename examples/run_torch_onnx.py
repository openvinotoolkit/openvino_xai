# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from openvino_xai import Task, insert_xai
from openvino_xai.common.utils import logger, softmax
from openvino_xai.explainer.visualizer import colormap, overlay

try:
    torch = importlib.import_module("torch")
    timm = importlib.import_module("timm")
except ImportError:
    logger.error("Please install torch and timm package to run this example.")
    exit(-1)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="resnet18.a1_in1k", type=str)
    parser.add_argument("--image_path", default="tests/assets/cheetah_person.jpg", type=str)
    parser.add_argument("--output_dir", default=".data/example", type=str)
    return parser


def run_insert_xai_torch(args: list[str]):
    """Insert XAI head into PyTorch model and run inference on PyTorch Runtime to get saliency map."""

    # Load Torch model from timm
    try:
        model = timm.create_model(args.model_name, in_chans=3, pretrained=True)
        logger.info(f"Model config: {model.default_cfg}")
        logger.info(f"Model layers: {model}")
    except Exception as e:
        logger.error(e)
        logger.info(f"Please choose from {timm.list_models()}")
        return
    input_size = model.default_cfg["input_size"][1:]  # (H, W)
    input_mean = np.array(model.default_cfg["mean"])
    input_std = np.array(model.default_cfg["std"])

    # Load image
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    image = cv2.resize(image, dsize=input_size)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image_norm = ((image/255.0 - input_mean)/input_std).astype(np.float32)
    image_norm = image_norm.transpose((2, 0, 1))  # HxWxC -> CxHxW
    image_norm = image_norm[None, :]  # CxHxW -> 1xCxHxW

    # Torch model inference
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(image_norm))
        probs = torch.softmax(logits, dim=-1)  # BxC
        label = probs.argmax(dim=-1)[0]
    logger.info(f"Torch model prediction: classes ({probs.shape[-1]}) -> label ({label}) -> prob ({probs[0, label]})")

    # Insert XAI head
    model_xai: torch.nn.Module = insert_xai(model, Task.CLASSIFICATION, input_size=input_size)  # Optional input size arg to help insertion

    # Torch XAI model inference
    model_xai.eval()
    with torch.no_grad():
        outputs = model_xai(torch.from_numpy(image_norm))
        logits = outputs["prediction"]  # BxC
        saliency_maps = outputs["saliency_map"]  # BxCxhxw
        probs = torch.softmax(logits, dim=-1)
        label = probs.argmax(dim=-1)[0]
    logger.info(f"Torch XAI model prediction: classes ({probs.shape[-1]}) -> label ({label}) -> prob ({probs[0, label]})")

    # Torch XAI model saliency map
    saliency_maps = saliency_maps.numpy(force=True).squeeze(0)  # Cxhxw
    saliency_map = saliency_maps[label]  # hxw saliency_map for the label
    saliency_map = colormap(saliency_map[None, :])  # 1xhxw
    saliency_map = cv2.resize(saliency_map.squeeze(0), dsize=input_size)  # HxW
    result_image = overlay(saliency_map, image)
    result_image = cv2.cvtColor(result_image, code=cv2.COLOR_RGB2BGR)
    result_image_path = Path(args.output_dir) / "xai-torch.png"
    result_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(result_image_path, result_image)
    logger.info(f"Torch XAI model saliency map: {result_image_path}")


def run_insert_xai_torch_to_onnx(args: list[str]):
    """Insert XAI head into PyTorch model, then converto to ONNX format and run inference on ONNX Runtime to get saliency map."""

    # ONNX import
    try:
        importlib.import_module("onnx")
        onnxruntime = importlib.import_module("onnxruntime")
    except ImportError:
        logger.info("Please install onnx and onnxruntime package to run ONNX XAI example.")
        return

    # Load Torch model from timm
    try:
        model = timm.create_model(args.model_name, in_chans=3, pretrained=True)
        logger.info(f"Model config: {model.default_cfg}")
        logger.info(f"Model layers: {model}")
    except Exception as e:
        logger.error(e)
        logger.info(f"Please choose from {timm.list_models()}")
        return
    input_size = model.default_cfg["input_size"][1:]  # (H, W)
    input_mean = np.array(model.default_cfg["mean"])
    input_std = np.array(model.default_cfg["std"])

    # Load image
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    image = cv2.resize(image, dsize=input_size)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image_norm = ((image/255.0 - input_mean)/input_std).astype(np.float32)
    image_norm = image_norm.transpose((2, 0, 1))  # HxWxC -> CxHxW
    image_norm = image_norm[None, :]  # CxHxW -> 1xCxHxW

    # Insert XAI head
    model_xai: torch.nn.Module = insert_xai(model, Task.CLASSIFICATION, input_size=input_size)

    # ONNX model conversion
    model_path = Path(args.output_dir) / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_xai,
        torch.from_numpy(image_norm),
        model_path,
        input_names=["input"],
        output_names=["prediction", "saliency_map"],
    )
    logger.info(f"ONNX XAI model: {model_path}")

    # ONNX model inference
    session = onnxruntime.InferenceSession(model_path)
    outputs = session.run(
        output_names=["prediction", "saliency_map"],
        input_feed={"input": image_norm.astype(np.float32)},
    )
    logits, saliency_maps = outputs  # NOTE: dict keys are removed in Torch->ONNX conversion
    probs = softmax(logits)
    label = probs.argmax(axis=-1)[0]
    logger.info(f"ONNX XAI model prediction: classes ({probs.shape[-1]}) -> label ({label}) -> prob ({probs[0, label]})")

    # ONNX model saliency map
    saliency_maps = saliency_maps.squeeze(0)  # Cxhxw
    saliency_map = saliency_maps[label]  # hxw saliency_map for the label
    saliency_map = colormap(saliency_map[None, :])  # 1xhxw
    saliency_map = cv2.resize(saliency_map.squeeze(0), dsize=input_size)  # HxW
    result_image = overlay(saliency_map, image)
    result_image = cv2.cvtColor(result_image, code=cv2.COLOR_RGB2BGR)
    result_image_path = Path(args.output_dir) / "xai-onnx.png"
    result_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(result_image_path, result_image)
    logger.info(f"ONNX XAI model saliency map: {result_image_path}")


def run_insert_xai_torch_to_openvino(args: list[str]):
    """Insert XAI head into PyTorch model, then convert to OpenVINO format and run inference on OpenVINO Runtime to get saliency map."""

    # Load Torch model from timm
    try:
        model = timm.create_model(args.model_name, in_chans=3, pretrained=True)
        logger.info(f"Model config: {model.default_cfg}")
        logger.info(f"Model layers: {model}")
    except Exception as e:
        logger.error(e)
        logger.info(f"Please choose from {timm.list_models()}")
        return
    input_size = model.default_cfg["input_size"][1:]  # (H, W)
    input_mean = np.array(model.default_cfg["mean"])
    input_std = np.array(model.default_cfg["std"])

    # Load image
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    image = cv2.resize(image, dsize=input_size)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image_norm = ((image/255.0 - input_mean)/input_std).astype(np.float32)
    image_norm = image_norm.transpose((2, 0, 1))  # HxWxC -> CxHxW
    image_norm = image_norm[None, :]  # CxHxW -> 1xCxHxW

    # Insert XAI head
    model_xai: torch.nn.Module = insert_xai(model, Task.CLASSIFICATION, input_size=input_size)

    # OpenVINO model conversion
    ov_model = ov.convert_model(
        model_xai,
        example_input=torch.from_numpy(image_norm),
        input=(ov.PartialShape([-1, *image_norm.shape[1:]],))
    )
    model_path = Path(args.output_dir) / "model.xml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, model_path)
    logger.info(f"OpenVINO XAI model: {model_path}")

    # OpenVINO XAI model inference
    ov_model = ov.Core().compile_model(ov_model, device_name="CPU")
    outputs = ov_model(image_norm)
    logits = outputs["prediction"]  # BxC
    saliency_maps = outputs["saliency_map"]  # BxCxhxw
    probs = softmax(logits)
    label = probs.argmax(axis=-1)[0]
    logger.info(f"OpenVINO XAI model prediction: classes ({probs.shape[-1]}) -> label ({label}) -> prob ({probs[0, label]})")

    # OpenVINO XAI model saliency map
    saliency_maps = saliency_maps.squeeze(0)  # Cxhxw
    saliency_map = saliency_maps[label]  # hxw saliency_map for the label
    saliency_map = colormap(saliency_map[None, :])  # 1xhxw
    saliency_map = cv2.resize(saliency_map.squeeze(0), dsize=input_size)  # HxW
    result_image = overlay(saliency_map, image)
    result_image = cv2.cvtColor(result_image, code=cv2.COLOR_RGB2BGR)
    result_image_path = Path(args.output_dir) / "xai-openvino.png"
    result_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(result_image_path, result_image)
    logger.info(f"OpenVINO XAI model saliency map: {result_image_path}")


def main(argv: list[str]):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    run_insert_xai_torch(args)
    run_insert_xai_torch_to_onnx(args)
    run_insert_xai_torch_to_openvino(args)


if __name__ == "__main__":
    main(sys.argv[1:])
