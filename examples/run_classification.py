# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.common.utils import logger
from openvino_xai.explainer.explainer import ExplainMode


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    x = x[:, :, ::-1]
    x = cv2.resize(src=x, dsize=(224, 224))
    x = x.transpose((2, 0, 1))
    x = np.expand_dims(x, 0)
    return x


def postprocess_fn(x) -> np.ndarray:
    return x["logits"]


def explain_auto(args):
    """
    Default use case using ExplainMode.AUTO.
    AUTO means that Explainer under the hood will attempt to use white-box methods and insert XAI branch in the model.
    If insertion fails, then black-box method will be applied.
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
    )

    # Prepare input image
    image = cv2.imread(args.image_path)

    # Generate explanation
    explanation = explainer(
        image,
        targets=[11, 14],  # target classes to explain
    )

    logger.info(
        f"explain_auto: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "explain_auto"
        explanation.save(output, f"{Path(args.image_path).stem}_")


def explain_white_box(args):
    """
    Advanced use case using ExplainMode.WHITEBOX.
    Insertion parameters (e.g. target_layer) are provided to further configure the white-box method (optional).
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
        explain_method=xai.Method.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
        # target_layer="last_conv_node_name",  # target_layer - node after which XAI branch will be inserted
        # target_layer="/backbone/conv/conv.2/Div",  # OTX mnet_v3
        # target_layer="/backbone/features/final_block/activate/Mul",  # OTX effnet
        embed_scaling=True, # True by default.  If set to True, saliency map scale (0 ~ 255) operation is embedded in the model
    )

    # Prepare input image and label names
    image = cv2.imread(args.image_path)
    voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # Generate explanation
    explanation = explainer(
        image,
        targets=[11, 14],  # target classes to explain, also ['dog', 'person'] is a valid input
        label_names=voc_labels,  # optional names
        overlay=True,
        )

    logger.info(
        f"explain_white_box: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "explain_white_box"
        explanation.save(output, f"{Path(args.image_path).stem}_")


def explain_black_box(args):
    """
    Advanced use case using ExplainMode.BLACKBOX.
    postprocess_fn is required for black-box methods.
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
    )

    # Prepare input image and label names
    image = cv2.imread(args.image_path)
    voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # Generate explanation
    explanation = explainer(
        image,
        targets=['dog', 'person'],  # target classes to explain, also [11, 14] possible
        label_names=voc_labels,  # optional names
        overlay=True,
    )

    logger.info(
        f"explain_black_box: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "explain_black_box"
        explanation.save(output, f"{Path(args.image_path).stem}_")


def explain_white_box_multiple_images(args):
    """
    Using the same explainer object to explain multiple images.
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
    )

    # Create list of images
    img_data_formats = (".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".png")
    if args.image_path.lower().endswith(img_data_formats):
        # args.image_path is a path to the image
        img_files = [args.image_path] * 5
    else:
        img_files = []
        # args.image_path is a directory (with sub-folder support)
        for root, _, _ in os.walk(args.image_path):
            for format_ in img_data_formats:
                img_files.extend([os.path.join(root, file.name) for file in Path(root).glob(f"*{format_}")])

    # Generate explanation
    images = [cv2.imread(image_path) for image_path in img_files]
    explanation = [explainer(image, targets=[14]) for image in images]

    logger.info(
        f"explain_white_box_multiple_images: Generated {len(explanation)} explanations "
        f"of layout {explanation[0].layout} with shape {explanation[0].shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "explain_white_box_multiple_images"
        explanation[0].save(output, f"{Path(args.image_path).stem}_")


def explain_white_box_vit(args):
    """Vision transformer example."""

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,  # defaults to AUTO
        explain_method=xai.Method.VITRECIPROCAM,
        # target_layer="/layers.10/ffn/Add",  # OTX deit-tiny
        # target_layer="/blocks/blocks.10/Add_1",  # timm vit_base_patch8_224.augreg_in21k_ft_in1k
    )

    # Prepare input image
    image = cv2.imread(args.image_path)

    # Generate explanation
    explanation = explainer(
        image,
        targets=[0, 1, 2, 3, 4],  # target classes to explain
    )

    logger.info(
        f"explain_white_box_vit: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.shape}."
    )

    # Save saliency maps for visual inspection
    if args.output is not None:
        output = Path(args.output) / "explain_white_box_vit"
        explanation.save(output, f"{Path(args.image_path).stem}_")


def insert_xai(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR, thus IR has additional 'saliency_map' output.
    """

    # Create ov.Model
    model = ov.Core().read_model(args.model_path)

    # insert XAI branch
    model_xai = xai.insert_xai(
        model,
        task=xai.Task.CLASSIFICATION,
    )

    logger.info("insert_xai: XAI branch inserted into IR.")

    # ***** Downstream task: user's code that infers model_xai and picks "saliency_map" output *****

    return model_xai


def insert_xai_w_params(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR with insertion parameters (e.g. target_layer), thus, IR has additional 'saliency_map' output.
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # insert XAI branch
    model_xai = xai.insert_xai(
        model,
        task=xai.Task.CLASSIFICATION,
        explain_method=xai.Method.RECIPROCAM,
        target_layer="/backbone/conv/conv.2/Div",  # OTX mnet_v3
        # target_layer="/backbone/features/final_block/activate/Mul",  # OTX effnet
        embed_scaling=True,
    )

    logger.info("insert_xai_w_params: XAI branch inserted into IR with parameters.")

    # ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****

    return model_xai


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # Get explanation
    explain_auto(args)
    explain_white_box(args)
    explain_black_box(args)
    explain_white_box_multiple_images(args)
    # explain_white_box_vit(args)

    # Insert XAI branch into the model
    insert_xai(args)
    insert_xai_w_params(args)


if __name__ == "__main__":
    main(sys.argv[1:])
