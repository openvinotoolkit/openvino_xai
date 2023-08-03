import sys
import argparse

import cv2

from openvino_xai.explain import WhiteBoxExplainer
from openvino_xai.model import XAIDetectionModel
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

    # # OTX YOLOX
    # cls_head_output_node_names = [
    #     "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
    # ]
    # OTX ATSS
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
    explainer = WhiteBoxExplainer(model)
    explanations = explainer.explain(image)
    logger.info(f"Generated detection saliency maps with shape {explanations.shape}.")
    if args.output is not None:
        save_explanations(args.output, explanations)


if __name__ == "__main__":
    main(sys.argv[1:])
