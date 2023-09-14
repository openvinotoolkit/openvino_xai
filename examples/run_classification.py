import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from openvino.model_api.models import ClassificationModel
import openvino.runtime as ov

from openvino_xai.explain import ClassificationAutoExplainer, RISEExplainer, WhiteBoxExplainer
from openvino_xai.parameters import ClassificationExplainParametersWB, PostProcessParameters, XAIMethodType
from openvino_xai.saliency_map import TargetExplainGroup
from openvino_xai.model import XAIClassificationModel
from openvino_xai.utils import logger


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def run_example_wo_explain_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = XAIClassificationModel.create_model(args.model_path, model_type="Classification")
    explainer = WhiteBoxExplainer(model)
    explanation = explainer.explain(image, target_explain_group=TargetExplainGroup.PREDICTED_CLASSES)
    logger.info(
        f"Example w/o explain_parameters: generated classification saliency maps "
        f"of layout {explanation.layout} with shape {explanation.map.shape}."
    )
    if args.output is not None:
        output = os.path.join(args.output, "wo_explain_parameters")
        explanation.save(output, image_name)


def run_example_w_explain_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    explain_parameters = ClassificationExplainParametersWB(
        target_layer="/backbone/conv/conv.2/Div",  # OTX mnet_v3
        # target_layer="/backbone/features/final_block/activate/Mul",  # OTX effnet
        explain_method_type=XAIMethodType.RECIPROCAM,
    )
    model = XAIClassificationModel.create_model(
        args.model_path, model_type="Classification", explain_parameters=explain_parameters
    )
    explainer = WhiteBoxExplainer(model)
    explanation = explainer.explain(image)
    logger.info(
        f"Example w/ explain_parameters: generated classification saliency maps "
        f"of layout {explanation.layout} with shape {explanation.map.shape}."
    )
    if args.output is not None:
        output = os.path.join(args.output, "w_explain_parameters")
        explanation.save(output, image_name)


def run_example_w_postprocessing_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = XAIClassificationModel.create_model(args.model_path, model_type="Classification")
    explainer = WhiteBoxExplainer(model)
    post_processing_parameters = PostProcessParameters(overlay=True)
    explanation = explainer.explain(
        image,
        TargetExplainGroup.PREDICTED_CLASSES,
        post_processing_parameters=post_processing_parameters,
    )
    logger.info(
        f"Example w/ post_processing_parameters: generated classification saliency maps "
        f"of layout {explanation.layout} with shape {explanation.map.shape}."
    )
    if args.output is not None:
        output = os.path.join(args.output, "w_postprocessing_parameters")
        explanation.save(output, image_name)


def run_blackbox_w_postprocessing_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = ClassificationModel.create_model(
        args.model_path, model_type="Classification", configuration={"output_raw_scores": True}
    )
    explainer = RISEExplainer(model)
    post_processing_parameters = PostProcessParameters(
        overlay=True,
    )
    explanation = explainer.explain(
        image,
        TargetExplainGroup.PREDICTED_CLASSES,
        post_processing_parameters=post_processing_parameters,
    )
    logger.info(
        f"Example from BlackBox explainer w/ post_processing_parameters: generated classification saliency maps "
        f"of layout {explanation.layout} with shape {explanation.map.shape}."
    )
    if args.output is not None:
        output = os.path.join(args.output, "blackbox_w_postprocessing_parameters")
        explanation.save(output, image_name)


def run_auto_example(args):
    """Auto example - try to get explanation with white-box method, if fails - use black-box"""
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = ClassificationModel.create_model(args.model_path, model_type="Classification")
    auto_explainer = ClassificationAutoExplainer(model)
    explanation = auto_explainer.explain(image)
    logger.info(
        f"Auto example: generated classification saliency maps "
        f"of layout {explanation.layout} with shape {explanation.map.shape}."
    )
    if args.output is not None:
        output = os.path.join(args.output, "auto_example")
        explanation.save(output, image_name)


def run_multiple_image_example(args):
    # TODO: wrap it into the explainer.explain() and enable async inference
    img_data_formats = (
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".png",
    )
    # single image path
    if args.image_path.lower().endswith(img_data_formats):
        # args.image_path is a path to the image
        img_files = [args.image_path]
    else:
        img_files = []
        # args.image_path is a directory (with sub-folder support)
        for root, _, _ in os.walk(args.image_path):
            for format_ in img_data_formats:
                img_files.extend([os.path.join(root, file.name) for file in Path(root).glob(f"*{format_}")])
    model = XAIClassificationModel.create_model(args.model_path, model_type="Classification")
    explainer = WhiteBoxExplainer(model)
    post_processing_parameters = PostProcessParameters(normalize=True, overlay=True)
    for image_path in img_files:
        image = cv2.imread(image_path)
        image_name = Path(image_path).stem
        explanation = explainer.explain(
            image,
            TargetExplainGroup.PREDICTED_CLASSES,
            post_processing_parameters=post_processing_parameters,
        )
        logger.info(
            f"Example w/ multiple images to explain: generated classification saliency maps "
            f"of layout {explanation.layout} with shape {explanation.map.shape}"
        )
        if args.output is not None:
            output = os.path.join(args.output, "multiple_image_example")
            explanation.save(output, image_name)


def run_ir_model_update_wo_inference(args):
    """Embedding XAI into the model graph and save updated IR, no actual inference performed.
    User suppose to use his/her own inference pipeline to get explanations along with the regular model outputs."""
    if args.output is not None:
        output = os.path.join(args.output, "ir_model_update_wo_inference")
        model_with_xai = XAIClassificationModel.insert_xai_into_native_ir(args.model_path, output)
        logger.info(f"Model with XAI head saved to {output}")
    else:
        model_with_xai = XAIClassificationModel.insert_xai_into_native_ir(args.model_path)


def run_ir_model_update_w_custom_inference(args):
    """Embedding XAI into the model graph and using custom inference pipeline to get explanations."""
    # Create IR with XAI head inserted
    model_with_xai = XAIClassificationModel.insert_xai_into_native_ir(args.model_path)

    # Load the model
    core = ov.Core()
    compiled_model = core.compile_model(model_with_xai, "CPU")

    # Load and pre-process input image
    # The model expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename=args.image_path), code=cv2.COLOR_BGR2RGB)
    image_name = Path(args.image_path).stem
    # Resize to imagenet image shape.
    input_image = cv2.resize(src=image, dsize=(224, 224))
    # Reshape to model input shape.
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.expand_dims(input_image, 0)

    # Inference model and parce the result
    result_infer = compiled_model([input_image])
    logits = result_infer["logits"]
    result_index = np.argmax(logits)
    saliency_map = result_infer["saliency_map"]
    saliency_map_of_predicted_class_raw = saliency_map[0][result_index]

    # Post-process saliency map
    saliency_map = cv2.resize(saliency_map_of_predicted_class_raw, image.shape[:2][::-1])
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    saliency_map = image * 0.5 + saliency_map * 0.5
    saliency_map[saliency_map > 255] = 255
    saliency_map = saliency_map.astype(np.uint8)

    logger.info(
        f"Example w/ IR update and custom inference: "
        f"generated classification saliency maps with shape {saliency_map.shape}"
    )
    if args.output is not None:
        # Save saliency map
        output = os.path.join(args.output, "ir_model_update_w_custom_inference")
        os.makedirs(output, exist_ok=True)
        cv2.imwrite(os.path.join(output, f"{image_name}_class{result_index}.jpg"), img=saliency_map)


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # E2E model explanation examples: patching IR model with XAI branch and using ModelAPI as inference framework
    run_example_wo_explain_parameters(args)
    run_example_w_explain_parameters(args)
    run_example_w_postprocessing_parameters(args)
    run_blackbox_w_postprocessing_parameters(args)
    run_auto_example(args)
    run_multiple_image_example(args)

    # Embedding XAI into the model graph and save updated IR (no inference performed)
    run_ir_model_update_wo_inference(args)
    # To get explanations along with the regular model output, user suppose to use his/her own custom inference pipeline
    run_ir_model_update_w_custom_inference(args)


if __name__ == "__main__":
    main(sys.argv[1:])
