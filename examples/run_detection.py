import sys
import argparse
from pathlib import Path

import cv2
import openvino.model_api as mapi
from openvino.model_api.models import DetectionModel

import openvino_xai as ovxai
from openvino_xai.insertion.insertion_parameters import DetectionInsertionParameters
from openvino_xai.common.parameters import XAIMethodType
from openvino_xai.common.utils import logger


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def main(argv):
    """
    White-box scenario.
    Insertion of the XAI branch into the Model API wrapper, thus Model API wrapper has additional 'saliency_map' output.
    """

    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # Create openvino.model_api.models.Model
    mapi_wrapper = mapi.models.DetectionModel.create_model(
        args.model_path,
        model_type="ssd",
    )

    # Insert XAI branch into Model API wrapper
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
    insertion_parameters = DetectionInsertionParameters(
        target_layer=cls_head_output_node_names,
        num_anchors=[1, 1, 1, 1, 1],
        saliency_map_size=(23, 23),  # Optional
        explain_method_type=XAIMethodType.DETCLASSPROBABILITYMAP,  # Optional
    )
    # TODO: support also custom DetectionModelInferrer for simple cases (just for a reference to users)
    model_inferrer = ovxai.insertion.insert_xai_into_mapi_wrapper(
        mapi_wrapper, insertion_parameters=insertion_parameters
    )

    # Generate explanation
    image = cv2.imread(args.image_path)
    explanation = ovxai.explain(
        model_inferrer,
        image,
    )

    logger.info(
        f"Generated {len(explanation.saliency_map)} detection "
        f"saliency maps of layout {explanation.layout} with shape {explanation.sal_map_shape}."
    )

    # Save saliency maps
    if args.output is not None:
        output = Path(args.output) / "detection"
        explanation.save(output, Path(args.image_path).stem)


if __name__ == "__main__":
    main(sys.argv[1:])
