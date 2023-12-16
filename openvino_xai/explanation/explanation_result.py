# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

import openvino.model_api as mapi

from openvino_xai.common.utils import logger
from openvino_xai.explanation.explanation_parameters import TargetExplainGroup, SELECTED_TARGETS, SaliencyMapLayout
from openvino_xai.explanation.utils import InferenceResult, get_prediction_from_model_output, select_target_indices


class ExplanationResult:
    """
    ExplanationResult selects target saliency maps, holds it and its layout.

    :param inference_result: Raw inference result, that includes model predictions and saliency maps.
    :type inference_result: InferenceResult
    :param target_explain_group: Defines targets to explain: all, only predictions, custom list, per-image.
    :type target_explain_group: TargetExplainGroup
    :param custom_target_indices: List of custom targets, optional.
    :type custom_target_indices: Optional[List[int]]
    :param confidence_threshold: Prediction confidence threshold.
    :type confidence_threshold:  float
    :param explain_target_names: List of all explain_target_names.
    :type explain_target_names: List[str]
    """

    def __init__(
        self,
        inference_result: Union[InferenceResult, mapi.models.ClassificationResult, mapi.models.DetectionResult],
        target_explain_group: TargetExplainGroup = TargetExplainGroup.PREDICTIONS,
        explain_target_names: Optional[List[str]] = None,
        custom_target_indices: Optional[List[int]] = None,
        confidence_threshold: float = 0.5,
    ):
        if not isinstance(
            inference_result, (InferenceResult, mapi.models.ClassificationResult, mapi.models.DetectionResult)
        ):
            raise ValueError(
                f"Input result has to be ether "
                f"openvino_xai.explanation.utils.InferenceResult or "
                f"openvino.model_api.models.ClassificationResult, but got {type(inference_result)}."
            )

        self.saliency_map = self._get_saliency_map_from_model_output(inference_result)

        if "per_image_map" in self.saliency_map:
            self.layout = SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY
            if target_explain_group != TargetExplainGroup.IMAGE:
                logger.warning(
                    f"Setting target_explain_group to TargetExplainGroup.IMAGE, {target_explain_group} "
                    f"is not supported when only single (global) saliency map per image is available."
                )
            self.target_explain_group = TargetExplainGroup.IMAGE
        else:
            if target_explain_group == TargetExplainGroup.IMAGE:
                raise ValueError(
                    "TargetExplainGroup.IMAGE supports only single (global) saliency map per image. "
                    "But multiple saliency maps are available."
                )
            self.layout = SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
            self.target_explain_group = target_explain_group

        self.confidence_threshold = confidence_threshold
        if self.target_explain_group in SELECTED_TARGETS:
            self.prediction, self.prediction_raw = get_prediction_from_model_output(
                inference_result, self.confidence_threshold
            )
            self.prediction_indices = [prediction[0] for prediction in self.prediction]
            if self.target_explain_group == TargetExplainGroup.PREDICTIONS:
                if not self.prediction:
                    raise ValueError(
                        "TargetExplainGroup.PREDICTIONS requires predictions "
                        "to be available, but currently model has no predictions. "
                        "Try to: (1) adjust preprocessing, (2) use different input, "
                        "(3) decrease confidence threshold, (4) retrain/re-export the model, etc."
                    )
            self.saliency_map = self._select_target_saliency_maps(custom_target_indices)

        self.explain_target_names = explain_target_names

    @property
    def sal_map_shape(self):
        idx = next(iter(self.saliency_map))
        sal_map_shape = self.saliency_map[idx].shape
        return sal_map_shape

    @classmethod
    def _get_saliency_map_from_model_output(cls, inference_result: InferenceResult):
        raw_saliency_map = inference_result.saliency_map
        if raw_saliency_map is None:
            raise RuntimeError("Inference result does not contain saliency_map.")
        if not isinstance(raw_saliency_map, np.ndarray):
            raise ValueError(f"Raw saliency_map has to be np.ndarray, but got {type(raw_saliency_map)}.")
        if raw_saliency_map.size == 0:
            raise RuntimeError("Inference result does not contain valid saliency_map.")
        if raw_saliency_map.shape[0] > 1:
            raise RuntimeError("Batch size for returned saliency maps should be 1.")

        saliency_map = cls._format_sal_map_as_dict(raw_saliency_map)
        return saliency_map

    @staticmethod
    def _format_sal_map_as_dict(raw_saliency_map: np.ndarray) -> Dict[Union[int, str], np.ndarray]:
        """Returns dict with saliency maps in format {target_id: class_saliency_map}."""
        dict_sal_map: Dict[Union[int, str], np.ndarray]
        if raw_saliency_map.ndim == 3:
            # Per-image saliency map
            dict_sal_map = {"per_image_map": raw_saliency_map[0]}
        elif raw_saliency_map.ndim == 4:
            # Per-target saliency map
            dict_sal_map = {}
            for index, sal_map in enumerate(raw_saliency_map[0]):
                dict_sal_map[index] = sal_map
        else:
            raise ValueError(
                f"Raw saliency map has to be tree or four dimensional tensor, " f"but got {raw_saliency_map.ndim}."
            )
        return dict_sal_map

    def _select_target_saliency_maps(
        self, custom_target_indices: Optional[List[int]] = None
    ) -> Dict[Union[int, str], np.ndarray]:
        assert self.layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
        explain_target_indexes = select_target_indices(
            self.target_explain_group,
            self.prediction_indices,
            custom_target_indices,
            len(self.saliency_map),
        )
        saliency_maps_selected = {i: self.saliency_map[i] for i in explain_target_indexes}
        return saliency_maps_selected

    def save(self, dir_path: Union[Path, str], name: Optional[str] = None) -> None:
        """Dumps saliency map."""
        # TODO: add unit test
        os.makedirs(dir_path, exist_ok=True)
        save_name = f"{name}_" if name else ""
        for idx, map_to_save in self.saliency_map.items():
            if idx == "per_image_map":
                target_name = "per_image_map"
            else:
                if self.explain_target_names:
                    target_name = self.explain_target_names[idx]
                else:
                    target_name = idx
            cv2.imwrite(os.path.join(dir_path, f"{save_name}target_{target_name}.jpg"), img=map_to_save)
