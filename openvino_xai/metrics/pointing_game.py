# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple

import numpy as np

from openvino_xai.common.utils import logger
from openvino_xai.explainer.explanation import ONE_MAP_LAYOUTS, Explanation
from openvino_xai.metrics.base import BaseMetric


class PointingGame(BaseMetric):
    """
    Implementation of the Pointing Game by Zhang et al., 2018.

    Unlike the original approach that uses ground truth bounding masks, this implementation uses ground
    truth bounding boxes. The Pointing Game checks whether the most salient point is within the annotated
    object. High scores mean that the most salient pixel belongs to an object of the specified class.

    References:
        1) Reference implementation:
           https://github.com/understandable-machine-intelligence-lab/Quantus/
           Hedström, Anna, et al.:
           "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond."
           Journal of Machine Learning Research 24.34 (2023): 1-11.
        2) Jianming Zhang et al.:
           "Top-Down Neural Attention by Excitation Backprop." International Journal of Computer Vision
           (2018) 126:1084-1102.
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(saliency_map: np.ndarray, image_gt_bboxes: List[Tuple[int, int, int, int]]) -> Dict[str, float]:
        """
        Calculate the Pointing Game metric for one saliency map for one class.

        This implementation uses a saliency map and bounding boxes of the same image and class.
        Returns a dictionary with the result of the Pointing Game metric.
        1.0 if any of the most salient points fall within the ground truth bounding boxes, 0.0 otherwise.

        :param saliency_map: A 2D numpy array representing the saliency map for the image.
        :type saliency_map: np.ndarray
        :param image_gt_bboxes: A list of tuples (x, y, w, h) representing the bounding boxes of the ground truth objects.
        :type image_gt_bboxes: List[Tuple[int, int, int, int]]

        :return: A dictionary with the result of the Pointing Game metric.
        :rtype: Dict[str, float]
        """
        # TODO: Optimize calculation by generating a mask from annotation and finding the intersection
        # Find the most salient points in the saliency map
        max_indices = np.argwhere(saliency_map == np.max(saliency_map))

        # If multiple bounding boxes are available for one image
        for x, y, w, h in image_gt_bboxes:
            for max_point_y, max_point_x in max_indices:
                # Check if this point is within the ground truth bounding box
                if x <= max_point_x <= x + w and y <= max_point_y <= y + h:
                    return {"pointing_game": 1.0}
        return {"pointing_game": 0.0}

    def evaluate(
        self,
        explanations: List[Explanation],
        gt_bboxes: List[Dict[str, List[Tuple[int, int, int, int]]]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Evaluates the Pointing Game metric over a set of images. Skips saliency maps if the gt bboxes for this class are absent.

        :param explanations: A list of explanations for each image.
        :type explanations: List[Explanation]
        :param gt_bboxes: A list of dictionaries {label_name: lists of bounding boxes} for each image.
        :type gt_bboxes: List[Dict[str, List[Tuple[int, int, int, int]]]]

        :return: Dict with "Pointing game" as a key and score over a list of images as as a value.
        :rtype: Dict[str, float]
        """

        assert len(explanations) == len(
            gt_bboxes
        ), "Number of explanations and ground truth bounding boxes must match and equal to number of images."

        hits = 0.0
        num_sal_maps = 0
        for explanation, image_gt_bboxes in zip(explanations, gt_bboxes):
            for class_idx, class_sal_map in explanation.saliency_map.items():
                if explanation.layout in ONE_MAP_LAYOUTS:
                    # Activation map
                    class_gt_bboxes = [
                        gt_bbox for class_gt_bboxes in image_gt_bboxes.values() for gt_bbox in class_gt_bboxes
                    ]
                else:
                    label_names = explanation.label_names
                    assert label_names is not None, "Label names are required for pointing game evaluation."
                    label_name = label_names[int(class_idx)]

                    if label_name not in image_gt_bboxes:
                        logger.info(
                            f"No ground-truth bbox for {label_name} saliency map. "
                            f"Skip pointing game evaluation for this saliency map."
                        )
                        continue
                    class_gt_bboxes = image_gt_bboxes[label_name]

                hits += self(class_sal_map, class_gt_bboxes)["pointing_game"]
                num_sal_maps += 1

        score = hits / num_sal_maps if num_sal_maps > 0 else 0.0
        return {"pointing_game": score}
