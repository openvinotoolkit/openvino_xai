# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
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
from openvino_xai.metrics import ADCC, InsertionDeletionAUC, PointingGame
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

datasets = pytest.importorskip("torchvision.datasets")


class DatasetType(Enum):
    COCO = "coco"
    VOC = "voc"


def coco_anns_to_gt_bboxes(
    anns: List[Dict[str, Any]] | Dict[str, Any], coco_val_labels: Dict[int, str]
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    gt_bboxes = {}
    for ann in anns:
        category_id = ann["category_id"]
        category_name = coco_val_labels[category_id]
        bbox = ann["bbox"]
        if category_name not in gt_bboxes:
            gt_bboxes[category_name] = []
        gt_bboxes[category_name].append(bbox)
    return gt_bboxes


def voc_anns_to_gt_bboxes(
    anns: List[Dict[str, Any]] | Dict[str, Any], *args: Any
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    gt_bboxes = {}
    anns = anns["annotation"]["object"]
    for ann in anns:
        category_name = ann["name"]
        bndbox = list(map(float, ann["bndbox"].values()))
        bndbox = np.array(bndbox, dtype=np.int32)
        x_min, y_min, x_max, y_max = bndbox
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        if category_name not in gt_bboxes:
            gt_bboxes[category_name] = []
        gt_bboxes[category_name].append(bbox)
    return gt_bboxes


def define_dataset_type(data_root: str, ann_path: str) -> DatasetType:
    if data_root and ann_path and ann_path.lower().endswith(".json"):
        if any(image_name.endswith(".jpg") for image_name in os.listdir(data_root)):
            return DatasetType.COCO

    required_voc_dirs = {"JPEGImages", "SegmentationObject", "ImageSets", "Annotations", "SegmentationClass"}
    for _, dir, _ in os.walk(data_root):
        if required_voc_dirs.issubset(set(dir)):
            return DatasetType.VOC

    raise ValueError("Dataset type is not supported")


@pytest.mark.parametrize(
    "data_root, ann_path",
    [
        ("tests/assets/cheetah_coco/images/val", "tests/assets/cheetah_coco/annotations/instances_val.json"),
        ("tests/assets/cheetah_voc", None),
    ],
)
class TestAccuracy:
    MODEL_NAME = "mlc_mobilenetv3_large_voc"

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root, data_root, ann_path):
        data_dir = fxt_data_root
        retrieve_otx_model(data_dir, self.MODEL_NAME)
        model_path = data_dir / "otx_models" / (self.MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        self.setup_dataset(data_root, ann_path)

        self.preprocess_fn = get_preprocess_fn(
            change_channel_order=self.channel_format == "BGR",
            input_size=(224, 224),
            hwc_to_chw=True,
        )
        self.postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)

        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

        self.pointing_game = PointingGame()
        self.auc = InsertionDeletionAUC(model, self.preprocess_fn, self.postprocess_fn)
        self.adcc = ADCC(model, self.preprocess_fn, self.postprocess_fn, self.explainer)

    def setup_dataset(self, data_root: str, ann_path: str):
        self.dataset_type = define_dataset_type(data_root, ann_path)
        self.channel_format = "RGB" if self.dataset_type in [DatasetType.VOC, DatasetType.COCO] else "None"

        if self.dataset_type == DatasetType.COCO:
            self.dataset = datasets.CocoDetection(root=data_root, annFile=ann_path)
            self.dataset_labels_dict = {cats["id"]: cats["name"] for cats in self.dataset.coco.cats.values()}
            self.anns_to_gt_bboxes = coco_anns_to_gt_bboxes
        elif self.dataset_type == DatasetType.VOC:
            self.dataset = datasets.VOCDetection(root=data_root, download=False, year="2012", image_set="val")
            self.dataset_labels_dict = None
            self.anns_to_gt_bboxes = voc_anns_to_gt_bboxes

    def test_explainer_images(self):
        images, explanations, dataset_gt_bboxes = [], [], []
        for image, anns in self.dataset:
            image_np = np.array(image)
            gt_bbox_dict = self.anns_to_gt_bboxes(anns, self.dataset_labels_dict)
            targets = [target for target in gt_bbox_dict.keys() if target in VOC_NAMES]

            explanation = self.explainer(image_np, targets=targets, label_names=VOC_NAMES, colormap=False)

            images.append(image_np)
            explanations.append(explanation)
            dataset_gt_bboxes.append({key: value for key, value in gt_bbox_dict.items() if key in targets})

        pointing_game = self.pointing_game.evaluate(explanations, dataset_gt_bboxes)
        auc = self.auc.evaluate(explanations, images, steps=10)
        adcc = self.adcc.evaluate(explanations, images)

        return {**pointing_game, **auc, **adcc}
