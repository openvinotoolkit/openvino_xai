# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.common.utils import logger
from openvino_xai.explainer.explainer import ExplainMode
from openvino_xai.methods.black_box.base import Preset


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # TODO: make sure it is correct
    # x = cv2.resize(src=x, dsize=(416, 416))  # OTX YOLOX
    x = cv2.resize(src=x, dsize=(992, 736))  # OTX ATSS
    x = x.transpose((2, 0, 1))
    x = np.expand_dims(x, 0)
    return x


def postprocess_fn(x) -> np.ndarray:
    """Returns boxes, scores, labels."""
    return x["boxes"][:, :, :4], x["boxes"][:, :, 4], x["labels"]


def explain_white_box(args):
    """
    White-box scenario.
    Per-class saliency map generation for single-stage detection models (using DetClassProbabilityMap).
    Insertion of the XAI branch into the model, thus model has additional 'saliency_map' output.
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # # OTX YOLOX
    # cls_head_output_node_names = [
    #     "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
    # ]
    # OTX ATSS
    cls_head_output_node_names = [
        "/bbox_head/atss_cls_1/Conv/WithoutBiases",
        "/bbox_head/atss_cls_2/Conv/WithoutBiases",
        "/bbox_head/atss_cls_3/Conv/WithoutBiases",
        "/bbox_head/atss_cls_4/Conv/WithoutBiases",
    ]

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.DETECTION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
        target_layer=cls_head_output_node_names,
        saliency_map_size=(23, 23),  # Optional
    )

    # Prepare input image and explanation parameters, can be different for each explain call
    image = cv2.imread(args.image_path)

    # Generate explanation
    explanation = explainer(
        image,
        targets=[0, 1, 2],  # target classes to explain
        overlay=True,
    )

    logger.info(
        f"Generated {len(explanation.saliency_map)} detection "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "detection"
        explanation.save(output, Path(args.image_path).stem)


def explain_black_box(args):
    """
    Black-box scenario.
    Per-box saliency map generation for all detection models (using AISEDetection).
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.DETECTION,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
    )

    # Prepare input image and explanation parameters, can be different for each explain call
    image = cv2.imread(args.image_path)

    # Generate explanation
    explanation = explainer(
        image,
        targets=[0],  # target boxes to explain
        overlay=True,
        preset=Preset.SPEED,
    )

    logger.info(
        f"Generated {len(explanation.saliency_map)} detection "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "detection_black_box"
        explanation.save(output, f"{Path(args.image_path).stem}_")


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    explain_white_box(args)
    explain_black_box(args)


if __name__ == "__main__":
    main(sys.argv[1:])
