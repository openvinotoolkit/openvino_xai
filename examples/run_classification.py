import os
import sys
import argparse
from pathlib import Path

import cv2
import openvino
from openvino.model_api.models import ClassificationModel
import openvino.runtime as ov

import openvino_xai as ovxai
from openvino_xai.common.parameters import TaskType, XAIMethodType
from openvino_xai.explanation.explanation_parameters import ExplainMode, PostProcessParameters, TargetExplainGroup, \
    ExplanationParameters
from openvino_xai.explanation.model_inferrer import ClassificationModelInferrer, ActivationType
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters
from openvino_xai.common.utils import logger


# USE_CUSTOM_INFERRER - if True, use provided custom model inference pipeline,
# otherwise, use Model API wrapper for inference.
USE_CUSTOM_INFERRER = True


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None, type=str)
    return parser


def insert_xai(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR, thus IR has additional 'saliency_map' output.
    """

    # Create ov.Model
    model = ov.Core().read_model(args.model_path)

    # insert XAI branch
    model_xai = ovxai.insert_xai(
        model,
        task_type=TaskType.CLASSIFICATION,
    )

    logger.info(f"insert_xai: XAI branch inserted into IR.")

    # ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****

    return model_xai


def insert_xai_w_params(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR with insertion parameters, thus IR has additional 'saliency_map' output.
    """

    # Create ov.Model
    model: ov.Model
    model = ov.Core().read_model(args.model_path)

    # Define insertion parameters
    insertion_parameters = ClassificationInsertionParameters(
        target_layer="/backbone/conv/conv.2/Div",  # OTX mnet_v3
        # target_layer="/backbone/features/final_block/activate/Mul",  # OTX effnet
        embed_normalization=True,
        explain_method_type=XAIMethodType.RECIPROCAM,
    )

    # insert XAI branch
    model_xai = ovxai.insert_xai(
        model,
        task_type=TaskType.CLASSIFICATION,
        insertion_parameters=insertion_parameters,
    )

    logger.info(f"insert_xai_w_params: XAI branch inserted into IR with parameters.")

    # ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****

    return model_xai


def insert_xai_into_mapi_wrapper(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the Model API wrapper, thus Model API wrapper has additional 'saliency_map' output.
    """

    # Create openvino.model_api.models.Model
    mapi_wrapper: openvino.model_api.models.Model
    mapi_wrapper = openvino.model_api.models.ClassificationModel.create_model(
        args.model_path,
        model_type="Classification",
    )

    # insert XAI branch into Model API wrapper
    mapi_wrapper_xai: openvino.model_api.models.Model
    mapi_wrapper_xai = ovxai.insertion.insert_xai_into_mapi_wrapper(mapi_wrapper)

    logger.info(f"insert_xai_into_mapi_wrapper: XAI branch inserted into Model API wrapper.")

    # ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****

    return mapi_wrapper_xai


def insert_xai_and_explain(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR, thus IR has additional 'saliency_map' output.
    Definition of a callable model_inferrer.
    Generate explanation.
    Save saliency maps.
    """

    # insert XAI branch into IR
    model_xai = insert_xai(args)

    # ***** Start of user's code that creates a callable model_inferrer *****
    # inference_result = model_inferrer(image)  # inference_result: ovxai.explanation.utils.InferenceResult
    if USE_CUSTOM_INFERRER:
        model_inferrer = ovxai.explanation.model_inferrer.ClassificationModelInferrer(
            model_xai, change_channel_order=True, activation=ActivationType.SIGMOID
        )
    else:
        model_inferrer = insert_xai_into_mapi_wrapper(args)
    # ***** End of user's code that creates a callable model_inferrer *****

    # Generate explanation
    image = cv2.imread(args.image_path)
    explanation = ovxai.explain(
        model_inferrer,
        image,
    )

    logger.info(
        f"insert_xai_and_explain: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.sal_map_shape}."
    )

    # Save saliency maps
    if args.output is not None:
        output = Path(args.output) / "explain"
        explanation.save(output, Path(args.image_path).stem)


def insert_xai_into_vit_and_explain(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR, thus IR has additional 'saliency_map' output.
    Definition of a callable model_inferrer.
    Generate explanation.
    Save saliency maps.
    """
    # Create ov.Model
    model = ov.Core().read_model(args.model_path)

    # Define insertion parameters
    insertion_parameters = ClassificationInsertionParameters(
        # target_layer="/layers.10/ffn/Add",  # OTX deit-tiny
        # target_layer="/blocks/blocks.10/Add_1",  # timm vit_base_patch8_224.augreg_in21k_ft_in1k
        explain_method_type=XAIMethodType.VITRECIPROCAM,
    )

    # insert XAI branch
    model_xai = ovxai.insert_xai(
        model,
        task_type=TaskType.CLASSIFICATION,
        insertion_parameters=insertion_parameters,
    )
    model_inferrer = ovxai.explanation.model_inferrer.ClassificationModelInferrer(
        model_xai, change_channel_order=True, activation=ActivationType.NONE
    )
    image = cv2.imread(args.image_path)
    explanation = ovxai.explain(
        model_inferrer,
        image,
    )

    logger.info(
        f"insert_xai_into_vit_and_explain: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.sal_map_shape}."
    )

    # Save saliency maps
    if args.output is not None:
        output = Path(args.output) / "vit"
        explanation.save(output, Path(args.image_path).stem)


def insert_xai_and_explain_w_params(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR, thus IR has additional 'saliency_map' output.
    Definition of a callable model_inferrer.
    Generate explanation.
    Save saliency maps.
    """

    # insert XAI branch into IR
    model_xai = insert_xai(args)

    # ***** Start of user's code that creates a callable model_inferrer *****
    # inference_result = model_inferrer(image)  # inference_result: ovxai.explanation.utils.InferenceResult
    if USE_CUSTOM_INFERRER:
        model_inferrer = ovxai.explanation.model_inferrer.ClassificationModelInferrer(
            model_xai, change_channel_order=True, activation=ActivationType.SIGMOID
        )
    else:
        model_inferrer = insert_xai_into_mapi_wrapper(args)
    # ***** End of user's code that creates a callable model_inferrer *****

    # Create explanation_parameters (optional)
    voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    explanation_parameters = ExplanationParameters(
        explain_mode=ExplainMode.WHITEBOX,  # by default, run white-box XAI
        target_explain_group=TargetExplainGroup.PREDICTIONS,  # by default, explains only predicted classes
        post_processing_parameters=PostProcessParameters(overlay=True),  # by default, saliency map overlays over image
        explain_target_names=voc_labels,  # optional
        confidence_threshold=0.6,  # defaults to 0.5
    )

    # Generate explanation
    image = cv2.imread(args.image_path)
    explanation = ovxai.explain(
        model_inferrer,
        image,
        explanation_parameters=explanation_parameters,
    )

    logger.info(
        f"insert_xai_and_explain_w_params: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.sal_map_shape}."
    )

    # Save saliency maps
    if args.output is not None:
        output = Path(args.output) / "explain_w_parameters"
        explanation.save(output, Path(args.image_path).stem)


def insert_xai_and_explain_multiple_images(args):
    """
    White-box scenario.
    Insertion of the XAI branch into the IR, thus IR has additional 'saliency_map' output.
    Definition of a callable model_inferrer.
    Generate explanation, per-image.
    Save saliency maps, per-image.
    """

    # TODO: wrap it into ovxai.explain()?

    # Create list of images
    img_data_formats = (".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".png")
    if args.image_path.lower().endswith(img_data_formats):
        # args.image_path is a path to the image
        img_files = [args.image_path]
    else:
        img_files = []
        # args.image_path is a directory (with sub-folder support)
        for root, _, _ in os.walk(args.image_path):
            for format_ in img_data_formats:
                img_files.extend([os.path.join(root, file.name) for file in Path(root).glob(f"*{format_}")])

    # insert XAI branch into IR
    model_xai = insert_xai(args)

    # ***** Start of user's code that creates a callable model_inferrer *****
    # inference_result = model_inferrer(image)  # inference_result: ovxai.explanation.utils.InferenceResult
    if USE_CUSTOM_INFERRER:
        model_inferrer = ovxai.explanation.model_inferrer.ClassificationModelInferrer(
            model_xai, change_channel_order=True, activation=ActivationType.SIGMOID
        )
    else:
        model_inferrer = insert_xai_into_mapi_wrapper(args)
    # ***** End of user's code that creates a callable model_inferrer *****

    # Create explanation_parameters (optional)
    post_processing_parameters = PostProcessParameters(overlay=True)
    explanation_parameters = ExplanationParameters(
        post_processing_parameters=post_processing_parameters,
    )

    for image_path in img_files:
        image = cv2.imread(image_path)
        explanation = ovxai.explain(
            model_inferrer,
            image,
            explanation_parameters=explanation_parameters,
        )

        logger.info(
            f"insert_xai_and_explain_multiple_images: generated {len(explanation.saliency_map)} classification "
            f"saliency maps of layout {explanation.layout} with shape {explanation.sal_map_shape}."
        )

        if args.output is not None:
            output = Path(args.output) / "multiple_images"
            explanation.save(output, Path(image_path).stem)


def explain_black_box(args):
    """
    Black-box scenario.
    Definition of a callable model_inferrer.
    Generate explanation in black-box mode.
    Save saliency maps.
    """

    # Create ov.Model
    model = ov.Core().read_model(args.model_path)

    # ***** Start of user's code that creates a callable model_inferrer *****
    # inference_result = model_inferrer(image)  # inference_result: ovxai.explanation.utils.InferenceResult
    if USE_CUSTOM_INFERRER:
        model_inferrer = ovxai.explanation.model_inferrer.ClassificationModelInferrer(
            model, change_channel_order=True, activation=ActivationType.SIGMOID
        )
    else:
        model_inferrer = openvino.model_api.models.ClassificationModel.create_model(
            args.model_path, model_type="Classification",  configuration={"output_raw_scores": True}
        )
    # ***** End of user's code that creates a callable model_inferrer *****

    # Create explanation_parameters
    explanation_parameters = ExplanationParameters(
        explain_mode=ExplainMode.BLACKBOX,
    )

    # Generate explanation
    image = cv2.imread(args.image_path)
    explanation = ovxai.explain(
        model_inferrer,
        image,
        explanation_parameters=explanation_parameters,
    )

    logger.info(
        f"explain_black_box: Generated {len(explanation.saliency_map)} classification "
        f"saliency maps of layout {explanation.layout} with shape {explanation.sal_map_shape}."
    )

    # Save saliency maps
    if args.output is not None:
        output = Path(args.output) / "black_box"
        explanation.save(output, Path(args.image_path).stem)


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # Insert XAI branch
    insert_xai(args)
    insert_xai_w_params(args)
    insert_xai_into_mapi_wrapper(args)

    # Insert XAI branch + get explanation in white-box mode
    insert_xai_and_explain(args)
    insert_xai_and_explain_w_params(args)
    insert_xai_and_explain_multiple_images(args)
    # insert_xai_into_vit_and_explain(args)

    # Get explanation in black-box mode
    explain_black_box(args)


if __name__ == "__main__":
    main(sys.argv[1:])
