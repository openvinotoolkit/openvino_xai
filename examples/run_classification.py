import os
import sys
import argparse
from pathlib import Path

import cv2
from openvino.model_api.models import ClassificationModel

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer, RISEExplainer
from openvino_xai.parameters import ClassificationExplainParametersWB, PostProcessParameters, XAIMethodType
from openvino_xai.saliency_map import TargetExplainGroup
from openvino_xai.model import XAIClassificationModel
from openvino_xai.utils import logger


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('--output', default=None, type=str)
    return parser


def run_example_wo_explain_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = XAIClassificationModel.create_model(args.model_path, "Classification")
    explainer = WhiteBoxExplainer(model)
    explanation = explainer.explain(image, target_explain_group=TargetExplainGroup.PREDICTED_CLASSES)
    logger.info(f"Example w/o explain_parameters: generated classification saliency maps "
                f"of layout {explanation.layout} with shape {explanation.map.shape}.")
    if args.output is not None:
        explanation.save(args.output, image_name)


def run_example_w_explain_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    explain_parameters = ClassificationExplainParametersWB(
        target_layer="/backbone/features/final_block/activate/Mul",  # OTX effnet
        # target_layer="/backbone/conv/conv.2/Div",  # OTX mnet_v3
        explain_method_type=XAIMethodType.RECIPROCAM,
    )
    model = XAIClassificationModel.create_model(args.model_path, "Classification",
                                                explain_parameters=explain_parameters)
    explainer = WhiteBoxExplainer(model)
    explanation = explainer.explain(image)
    logger.info(f"Example w/ explain_parameters: generated classification saliency maps "
                f"of layout {explanation.layout} with shape {explanation.map.shape}.")
    if args.output is not None:
        explanation.save(args.output, image_name)


def run_example_w_postprocessing_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = XAIClassificationModel.create_model(args.model_path, "Classification")
    explainer = WhiteBoxExplainer(model)
    post_processing_parameters = PostProcessParameters(overlay=True)
    explanation = explainer.explain(
        image,
        TargetExplainGroup.PREDICTED_CLASSES,
        post_processing_parameters=post_processing_parameters,
    )
    logger.info(f"Example w/ post_processing_parameters: generated classification saliency maps "
                f"of layout {explanation.layout} with shape {explanation.map.shape}.")
    if args.output is not None:
        explanation.save(args.output, image_name)


def run_blackbox_w_postprocessing_parameters(args):
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = ClassificationModel.create_model(args.model_path, configuration={"output_raw_scores": True})
    explainer = RISEExplainer(model)
    post_processing_parameters = PostProcessParameters(
        overlay=True,
    )
    explanation = explainer.explain(
        image,
        TargetExplainGroup.PREDICTED_CLASSES,
        post_processing_parameters=post_processing_parameters,
    )
    logger.info(f"Example from BlackBox explainer w/ post_processing_parameters: generated classification saliency maps "
                f"of layout {explanation.layout} with shape {explanation.map.shape}.")
    if args.output is not None:
        explanation.save(args.output, image_name)


def run_auto_example(args):
    """Auto example - try to get explanation with white-box method, if fails - use black-box"""
    image = cv2.imread(args.image_path)
    image_name = Path(args.image_path).stem

    model = ClassificationModel.create_model(args.model_path, "Classification")
    auto_explainer = ClassificationAutoExplainer(model)
    explanation = auto_explainer.explain(image)
    logger.info(f"Auto example: generated classification saliency maps "
                f"of layout {explanation.layout} with shape {explanation.map.shape}.")
    if args.output is not None:
        explanation.save(args.output, image_name)


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
    model = XAIClassificationModel.create_model(args.model_path, "Classification")
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
        logger.info(f"Example w/ multiple images to explain: generated classification saliency maps "
                    f"of layout {explanation.layout} with shape {explanation.map.shape}")
        if args.output is not None:
            explanation.save(args.output, image_name)


def run_ir_model_update_wo_inference(args):
    """Embedding XAI into the model graph and save updated IR, no actual inference performed.
    User suppose to use his/her own inference pipeline to get explanations along with the regular model outputs."""
    model_with_xai = XAIClassificationModel.insert_xai_into_native_ir(args.model_path, args.output)
    logger.info(f"Model with XAI head saved to {args.output}")


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # e2e model explanation examples: patching IR model with XAI branch and using ModelAPI as inference framework
    run_example_wo_explain_parameters(args)
    run_example_w_explain_parameters(args)
    run_example_w_postprocessing_parameters(args)
    run_blackbox_w_postprocessing_parameters(args)
    run_auto_example(args)
    run_multiple_image_example(args)

    # Embedding XAI into the model graph and save updated IR, no actual inference performed
    # User suppose to use his/her own custom inference pipeline to get explanations along with the regular model outputs
    run_ir_model_update_wo_inference(args)

if __name__ == "__main__":
    main(sys.argv[1:])
