# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict, List, Tuple

import cv2
import openvino as ov
import pytest

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
)
from openvino_xai.metrics.insertion_deletion_auc import InsertionDeletionAUC
from openvino_xai.metrics.pointing_game import PointingGame
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

MODEL_NAME = "mlc_mobilenetv3_large_voc"
IMAGE_PATH = "tests/assets/cheetah_person.jpg"
COCO_ANN_PATH = "tests/assets/cheetah_person_coco.json"


def load_gt_bboxes(json_coco_path: str) -> List[Dict[str, List[Tuple[int, int, int, int]]]]:
    """
    Loads ground truth bounding boxes from a COCO format JSON file.
    Returns a list of dictionaries, where each dictionary corresponds to an image.
    The key is the label name and the value is a list of bounding boxes for certain image.
    """

    with open(json_coco_path, "r") as f:
        coco_anns = json.load(f)

    result = {}
    category_id_to_name = {category["id"]: category["name"] for category in coco_anns["categories"]}

    for annotation in coco_anns["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]

        category_name = category_id_to_name[category_id]
        if image_id not in result:
            result[image_id] = {}
        if category_name not in result[image_id]:
            result[image_id][category_name] = []

        result[image_id][category_name].append(bbox)

    return list(result.values())


class TestDummyRegression:
    image = cv2.imread(IMAGE_PATH)
    gt_bboxes = load_gt_bboxes(COCO_ANN_PATH)
    pointing_game = PointingGame()

    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )

    postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        data_dir = fxt_data_root
        retrieve_otx_model(data_dir, MODEL_NAME)
        model_path = data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)
        compiled_model = ov.Core().compile_model(model, "CPU")
        self.auc = InsertionDeletionAUC(compiled_model, self.preprocess_fn, self.postprocess_fn)

        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

    def test_explainer_image(self):
        explanation = self.explainer(self.image, targets=["person"], label_names=VOC_NAMES, colormap=False)
        assert len(explanation.saliency_map) == 1
        pointing_game_score = self.pointing_game.evaluate([explanation], self.gt_bboxes)["pointing_game"]
        assert pointing_game_score == 1.0

        explanation = self.explainer(self.image, targets=["person"], label_names=VOC_NAMES, colormap=False)
        assert len(explanation.saliency_map) == 1
        auc_score = self.auc.evaluate([explanation], [self.image], steps=10).values()
        insertion_auc_score, deletion_auc_score, delta_auc_score = auc_score
        assert insertion_auc_score >= 0.9
        assert deletion_auc_score >= 0.2
        assert delta_auc_score >= 0.7

        # Two classes for saliency maps
        explanation = self.explainer(self.image, targets=["person", "cat"], label_names=VOC_NAMES, colormap=False)
        assert len(explanation.saliency_map) == 2
        auc_score = self.auc.evaluate([explanation], [self.image], steps=10).values()
        insertion_auc_score, deletion_auc_score, delta_auc_score = auc_score
        assert insertion_auc_score >= 0.5
        assert deletion_auc_score >= 0.1
        assert delta_auc_score >= 0.35

    def test_explainer_images(self):
        images = [self.image, self.image]
        explanations = []
        for image in images:
            explanation = self.explainer(image, targets=["person"], label_names=VOC_NAMES, colormap=False)
            explanations.append(explanation)
        dataset_gt_bboxes = self.gt_bboxes * 2

        pointing_game_score = self.pointing_game.evaluate(explanations, dataset_gt_bboxes)["pointing_game"]
        assert pointing_game_score == 1.0

        auc_score = self.auc.evaluate(explanations, images, steps=10).values()
        insertion_auc_score, deletion_auc_score, delta_auc_score = auc_score
        assert insertion_auc_score >= 0.9
        assert deletion_auc_score >= 0.2
        assert delta_auc_score >= 0.7
