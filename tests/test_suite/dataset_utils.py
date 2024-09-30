# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class DatasetType(Enum):
    COCO = "COCO"
    ILSVRC = "ILSVRC"
    VOC = "VOC"


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


def define_dataset_type(data_root: Path, ann_path: Path) -> DatasetType:
    if data_root and ann_path and ann_path.suffix == ".json":
        if any(image_name.endswith(".jpg") for image_name in os.listdir(data_root)):
            return DatasetType.COCO

    required_voc_dirs = {"JPEGImages", "ImageSets", "Annotations"}
    required_ilsvrc_dirs = {"Data", "ImageSets", "Annotations"}
    for _, dir, _ in os.walk(data_root):
        if required_ilsvrc_dirs.issubset(set(dir)):
            return DatasetType.ILSVRC
        if required_voc_dirs.issubset(set(dir)):
            return DatasetType.VOC

    raise ValueError("Dataset type is not supported")
