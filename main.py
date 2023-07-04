import sys

import argparse

from openvino_xai.insert import InsertXAICls, InsertXAIDet
from openvino_xai.methods import DetClassProbabilityMapXAIMethod, ReciproCAMXAIMethod


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--save_path')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # # # Classification example
    # explain_params = {"target_layer": None}
    # explain_method = ReciproCAMXAIMethod(args.model_path, **explain_params)
    # print("\nOriginal model:\n", explain_method.model_ori)
    #
    # ir_with_xai_generator = InsertXAICls(explain_method)
    # model_with_xai = ir_with_xai_generator.generate_model_with_xai()
    # print("\nModel with XAI inserted:\n", model_with_xai)
    # if args.save_path:
    #     ir_with_xai_generator.serialize_model_with_xai(args.save_path)

    # Detection example
    # Define explain params for the specific model and XAI method
    # # YOLOX
    # cls_head_output_node_names = [
    #     "/bbox_head/multi_level_conv_cls.0/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.1/Conv/WithoutBiases",
    #     "/bbox_head/multi_level_conv_cls.2/Conv/WithoutBiases",
    # ]
    # ATSS
    cls_head_output_node_names = [
        "/bbox_head/atss_cls_0/Conv/WithoutBiases",
        "/bbox_head/atss_cls_1/Conv/WithoutBiases",
        "/bbox_head/atss_cls_2/Conv/WithoutBiases",
        "/bbox_head/atss_cls_3/Conv/WithoutBiases",
        "/bbox_head/atss_cls_4/Conv/WithoutBiases",
    ]
    explain_params = {
        "cls_head_output_node_names": cls_head_output_node_names,
        "num_anchors": [1, 1, 1, 1, 1],
        "saliency_map_size": (23, 23),  # Optional
    }
    explain_method = DetClassProbabilityMapXAIMethod(args.model_path, **explain_params)
    print("\nOriginal model:\n", explain_method.model_ori)

    ir_with_xai_generator = InsertXAIDet(explain_method)
    model_with_xai = ir_with_xai_generator.generate_model_with_xai()
    print("\nModel with XAI inserted:\n", model_with_xai)
    if args.save_path:
        ir_with_xai_generator.serialize_model_with_xai(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
