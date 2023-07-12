import sys
import argparse

import cv2
from openvino.model_api.models import ClassificationModel

from openvino_xai.explain import WhiteBoxExplainer, ClassificationAutoExplainer
from openvino_xai.model import XAIClassificationModel, XAIDetectionModel
from openvino_xai.utils import logger


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('--save_path')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # Classification example
    explain_parameters = {
        "explain_method_name": "reciprocam",  # Optional
        "target_layer": "/backbone/features/final_block/activate/Mul",  # effnet
        # "target_layer": "/backbone/conv/conv.2/Div",  # mnet_v3
    }
    model = XAIClassificationModel.create_model(args.model_path, "Classification",
                                                explain_parameters=explain_parameters)
    image = cv2.imread(args.image_path)
    explainer = WhiteBoxExplainer(model)
    explanations = explainer.explain(image)
    logger.info(f"Generated classification saliency maps with shape {explanations.shape}.")

    # Classification auto example
    explain_parameters = {
        "explain_method_name": "reciprocam",  # Optional
        "target_layer": "/backbone/features/final_block/activate/Mul",  # effnet
        # "target_layer": "/backbone/conv/conv.2/Div",  # mnet_v3
    }
    model = ClassificationModel.create_model(args.model_path, "Classification")
    auto_explainer = ClassificationAutoExplainer(model, explain_parameters)
    image = cv2.imread(args.image_path)
    explanations = auto_explainer.explain(image)
    logger.info(f"Generated classification saliency maps with shape {explanations.shape}.")
    a = 1

    # Detection example
    # # YOLOX
    # cls_head_output_node_names = [
    #     "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
    # ]
    # ATSS
    cls_head_output_node_names = (
        "/bbox_head/atss_cls_1/Conv/WithoutBiases",
        "/bbox_head/atss_cls_2/Conv/WithoutBiases",
        "/bbox_head/atss_cls_3/Conv/WithoutBiases",
        "/bbox_head/atss_cls_4/Conv/WithoutBiases",
    )
    explain_parameters = {
        "explain_method_name": "detclassprobabilitymap",
        "target_layer": cls_head_output_node_names,
        "num_anchors": (1, 1, 1, 1, 1),
        "saliency_map_size": (23, 23),  # Optional
    }
    model = XAIDetectionModel.create_model(args.model_path, model_type="ssd",
                                                explain_parameters=explain_parameters)
    image = cv2.imread(args.image_path)
    explainer = WhiteBoxExplainer(model)
    explanations = explainer.explain(image)
    logger.info(f"Generated detection saliency maps with shape {explanations.shape}.")


if __name__ == "__main__":
    main(sys.argv[1:])
