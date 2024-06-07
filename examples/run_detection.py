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
from openvino_xai.explainer.parameters import (
    ExplainMode,
    ExplanationParameters,
    TargetExplainGroup,
)
from openvino_xai.inserter.parameters import DetectionInsertionParameters


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # TODO: make sure it is correct
    x = cv2.resize(src=x, dsize=(416, 416))  # OTX YOLOX
    # x = cv2.resize(src=x, dsize=(992, 736))  # OTX ATSS
    x = x.transpose((2, 0, 1))
    x = np.expand_dims(x, 0)
    return x


def main(argv):
    """
    White-box scenario.
    Insertion of the XAI branch into the Model API wrapper, thus Model API wrapper has additional 'saliency_map' output.
    """

    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # OTX YOLOX
    cls_head_output_node_names = [
        "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
        "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
        "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
    ]
    # # OTX ATSS
    # cls_head_output_node_names = [
    #     "/bbox_head/atss_cls_1/Conv/WithoutBiases",
    #     "/bbox_head/atss_cls_2/Conv/WithoutBiases",
    #     "/bbox_head/atss_cls_3/Conv/WithoutBiases",
    #     "/bbox_head/atss_cls_4/Conv/WithoutBiases",
    # ]
    insertion_parameters = DetectionInsertionParameters(
        target_layer=cls_head_output_node_names,
        # num_anchors=[1, 1, 1, 1, 1],
        saliency_map_size=(23, 23),  # Optional
        explain_method=xai.Method.DETCLASSPROBABILITYMAP,  # Optional
    )

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.DETECTION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
        insertion_parameters=insertion_parameters,
    )

    # Prepare input image and explanation parameters, can be different for each explain call
    image = cv2.imread(args.image_path)
    explanation_parameters = ExplanationParameters(
        target_explain_group=TargetExplainGroup.CUSTOM,  # CUSTOM list of classes to explain, also ALL possible
        target_explain_labels=[0, 1, 2, 3, 4],  # target classes to explain
    )

    # Generate explanation
    explanation = explainer(image, explanation_parameters)

    logger.info(
        f"Generated {len(explanation.saliency_map)} detection "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "detection"
        explanation.save(output, Path(args.image_path).stem)


if __name__ == "__main__":
    main(sys.argv[1:])
