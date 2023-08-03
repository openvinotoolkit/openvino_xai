import sys
import argparse

import cv2
from openvino.model_api.models import ClassificationModel

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer
from openvino_xai.model import XAIClassificationModel
from openvino_xai.utils import logger, save_explanations


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('--output', default=None, type=str)
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    image = cv2.imread(args.image_path)

    # Example w/ explain_parameters
    explain_parameters = {
        "explain_method_name": "reciprocam",  # Optional
        "target_layer": "/backbone/features/final_block/activate/Mul",  # OTX effnet
        # "target_layer": "/backbone/conv/conv.2/Div",  # OTX mnet_v3
    }
    model = XAIClassificationModel.create_model(args.model_path, "Classification",
                                                explain_parameters=explain_parameters)
    explainer = WhiteBoxExplainer(model)
    explanations = explainer.explain(image)
    logger.info(f"Example w/ explain_parameters: "
                f"generated classification saliency maps with shape {explanations.shape}.")
    if args.output is not None:
        save_explanations(args.output, explanations)

    # Example w/o explain_parameters
    model = XAIClassificationModel.create_model(args.model_path, "Classification")
    explainer = WhiteBoxExplainer(model)
    explanations = explainer.explain(image)
    logger.info(f"Example w/o explain_parameters: "
                f"generated classification saliency maps with shape {explanations.shape}.")
    if args.output is not None:
        save_explanations(args.output, explanations)

    # Auto example (try to get explanations with white-box method, if fails - use black-box)
    model = ClassificationModel.create_model(args.model_path, "Classification")
    auto_explainer = ClassificationAutoExplainer(model)
    explanations = auto_explainer.explain(image)
    logger.info(f"Auto example: generated classification saliency maps with shape {explanations.shape}.")
    if args.output is not None:
        save_explanations(args.output, explanations)


if __name__ == "__main__":
    main(sys.argv[1:])
