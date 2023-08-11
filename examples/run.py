import sys
import argparse

import cv2

from openvino_xai.explain import RISEExplainer
from openvino_xai.model import BlackBoxModel
from openvino_xai.utils import logger, save_explanations


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # Classification example
    # explain_parameters = {
    #     "explain_method_name": "reciprocam",  # Optional
    #     "target_layer": "/backbone/features/final_block/activate/Mul",  # effnet
    #     # "target_layer": "/backbone/conv/conv.2/Div",  # mnet_v3
    # }
    # model = XAIClassificationModel.create_model(args.model_path, "Classification",
    #                                             explain_parameters=explain_parameters)
    # image = cv2.imread(args.image_path)
    # explainer = WhiteBoxExplainer(model)
    # explanations = explainer.explain(image)
    # logger.info(f"Generated classification saliency maps with shape {explanations.shape}.")
    # if args.output is not None:
    #     save_explanations(args.output, explanations)

    # # Classification example w/o target layer
    # model = XAIClassificationModel.create_model(args.model_path, "Classification")
    # image = cv2.imread(args.image_path)
    # explainer = WhiteBoxExplainer(model)
    # explanations = explainer.explain(image)
    # logger.info(f"Generated classification saliency maps with shape {explanations.shape}.")
    # if args.output is not None:
    #     save_explanations(args.output, explanations)

    # # Classification auto example
    # model = ClassificationModel.create_model(args.model_path, "Classification")
    # auto_explainer = ClassificationAutoExplainer(model)
    # image = cv2.imread(args.image_path)
    # explanations = auto_explainer.explain(image)
    # logger.info(f"Generated classification saliency maps with shape {explanations.shape}.")
    # if args.output is not None:
    #     save_explanations(args.output, explanations)

    # # Detection example
    # # # YOLOX
    # # cls_head_output_node_names = [
    # #     "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
    # #     "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
    # #     "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
    # # ]
    # # ATSS
    # cls_head_output_node_names = (
    #     "/bbox_head/atss_cls_1/Conv/WithoutBiases",
    #     "/bbox_head/atss_cls_2/Conv/WithoutBiases",
    #     "/bbox_head/atss_cls_3/Conv/WithoutBiases",
    #     "/bbox_head/atss_cls_4/Conv/WithoutBiases",
    # )
    # explain_parameters = {
    #     "explain_method_name": "detclassprobabilitymap",
    #     "target_layer": cls_head_output_node_names,
    #     "num_anchors": (1, 1, 1, 1, 1),
    #     "saliency_map_size": (23, 23),  # Optional
    # }
    # model = XAIDetectionModel.create_model(args.model_path, model_type="ssd",
    #                                             explain_parameters=explain_parameters)
    # image = cv2.imread("/home/negvet/training_extensions/otx-workspace-DETECTION-wgisd-atss/splitted_dataset/val/images/val/CDY_2015.jpg")
    # explainer = WhiteBoxExplainer(model)
    # explanations = explainer.explain(image)
    # logger.info(f"Generated detection saliency maps with shape {explanations.shape}.")
    # if args.output is not None:
    #     save_explanations(args.output, explanations)

    import numpy as np

    # Classification example black box
    model = BlackBoxModel.create_model(args.model_path, "Classification")
    logger.info("Created Model API wrapper.")
    image = cv2.imread(args.image_path)
    explainer = RISEExplainer(model, None, None, None)
    explanations = explainer.explain(image)
    logger.info(f"Generated classification saliency maps with shape {explanations.shape}.")
    if args.output is not None:
        save_explanations(args.output, explanations)
        weight = 0.5
        shape = image.shape[1], image.shape[0]
        for idx, class_map in enumerate(explanations[0]):
            overlay = image * weight + cv2.resize(class_map, shape) * (1 - weight)
            overlay[overlay > 255] = 255
            overlay = overlay.astype(np.uint8)
            cv2.imwrite(f"{args.output}/overlay_{idx}.jpg", overlay)
        cv2.imwrite(f"{args.output}/source.jpg", image)


if __name__ == "__main__":
    main(sys.argv[1:])
