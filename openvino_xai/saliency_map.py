# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from openvino.model_api.models import ClassificationResult

from openvino_xai.parameters import PostProcessParameters


class TargetExplainGroup(Enum):
    """
    Enum describes different target explain groups.

    Contains the following values:
        IMAGE - Global (single) saliency map per image.
        ALL_CLASSES - Saliency map per each class.
        PREDICTED_CLASSES - Saliency map per each predicted class.
        CUSTOM_CLASSES - Saliency map per each specified class.
        PREDICTED_BBOXES - Saliency map per each predicted bbox.
        CUSTOM_BBOXES - Saliency map per each custom bbox.
    """

    IMAGE = "image"
    ALL_CLASSES = "all_classes"
    PREDICTED_CLASSES = "predicted_classes"
    CUSTOM_CLASSES = "custom_classes"
    PREDICTED_BBOXES = "predicted_bboxes"
    CUSTOM_BBOXES = "custom_bboxes"


SELECTED_CLASSES = {
    TargetExplainGroup.PREDICTED_CLASSES,
    TargetExplainGroup.CUSTOM_CLASSES,
}
SELECTED_BBOXES = {
    TargetExplainGroup.PREDICTED_BBOXES,
    TargetExplainGroup.CUSTOM_BBOXES,
}


class SaliencyMapLayout(Enum):
    """
    Enum describes different saliency map layouts.

    Saliency map can have the following layout:
        ONE_MAP_PER_IMAGE_GRAY - BHW - one map per image
        ONE_MAP_PER_IMAGE_COLOR - BHWC - one map per image, colormapped
        MULTIPLE_MAPS_PER_IMAGE_GRAY - BNHW - multiple maps per image
        MULTIPLE_MAPS_PER_IMAGE_COLOR - BNHWC - multiple maps per image, colormapped
    """

    ONE_MAP_PER_IMAGE_GRAY = "one_map_per_image_gray"
    ONE_MAP_PER_IMAGE_COLOR = "one_map_per_image_color"
    MULTIPLE_MAPS_PER_IMAGE_GRAY = "MULTIPLE_MAPS_PER_IMAGE_GRAY"
    MULTIPLE_MAPS_PER_IMAGE_COLOR = "MULTIPLE_MAPS_PER_IMAGE_COLOR"


GRAY_LAYOUTS = {
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY,
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
}
COLOR_MAPPED_LAYOUTS = {
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR,
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
MULTIPLE_MAP_LAYOUTS = {
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
    SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
ONE_MAP_LAYOUTS = {
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY,
    SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR,
}


class ExplainResult:
    """
    ExplainResult selects target saliency maps, holds it and its layout.

    :param raw_result: Raw prediction of ModelAPI wrapper.
    :type raw_result: ClassificationResult
    :param target_explain_group: Defines targets to explain: all classes, only predicted classes, custom classes, etc.
    :type target_explain_group: TargetExplainGroup
    :param explain_targets: Provides list of custom targets, optional.
    :type explain_targets: Optional[List[int]]
    :param labels: List of all labels.
    :type labels: List[str]
    """

    # TODO: Separate for task type, e.g. create ExplainResult <- ExplainResultClassification, etc.

    def __init__(
        self,
        raw_result: ClassificationResult,
        target_explain_group: TargetExplainGroup,
        explain_targets: Optional[List[int]] = None,
        labels: List[str] = None,
        hierarchical_info: Dict = None,
    ):
        self._labels = labels
        raw_saliency_map = self._get_saliency_map_from_predictions(raw_result)
        dict_sal_map = self._format_sal_map_as_dict(raw_saliency_map)

        self._layout = self.get_layout(dict_sal_map)
        if hierarchical_info and self._layout in MULTIPLE_MAP_LAYOUTS:
            dict_sal_map = self.reorder_hierarch_sal_map(dict_sal_map, hierarchical_info, labels)
        self._saliency_map = self._select_target_saliency_maps(
            dict_sal_map, target_explain_group, raw_result, explain_targets
        )
        if self._layout in MULTIPLE_MAP_LAYOUTS:
            self._predictions = raw_result.top_labels

    @property
    def map(self):
        return self._saliency_map

    @map.setter
    def map(self, saliency_map: Dict[int, np.ndarray]):
        self._saliency_map = saliency_map

    @property
    def sal_map_shape(self):
        idx = next(iter(self._saliency_map))
        sal_map_shape = self._saliency_map[idx].shape
        return sal_map_shape

    @property
    def predictions(self):
        return self._predictions

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout

    @staticmethod
    def _get_saliency_map_from_predictions(raw_result: ClassificationResult):
        raw_saliency_map = raw_result.saliency_map
        if raw_saliency_map.size == 0:
            raise RuntimeError("Model does not contain saliency_map output.")
        if raw_saliency_map.shape[0] > 1:
            raise RuntimeError("Batch size for returned saliency maps should be 1.")
        return raw_saliency_map

    @staticmethod
    def _format_sal_map_as_dict(raw_saliency_map: np.ndarray) -> Dict[Union[int, str], np.ndarray]:
        """Returns dict with saliency maps in format {target_id: class_saliency_map}."""
        if raw_saliency_map.ndim == 3:
            # Per-image saliency map
            dict_sal_map = {"per_image_map": raw_saliency_map[0]}
        elif raw_saliency_map.ndim == 4:
            # Per-class saliency map
            dict_sal_map = {}
            for idx, class_sal in enumerate(raw_saliency_map[0]):
                dict_sal_map[idx] = class_sal
        else:
            raise ValueError(
                f"Raw saliency map has to be tree or four dimensional tensor, "
                f"but got {raw_saliency_map.ndim}."
            )
        return dict_sal_map

    @staticmethod
    def _check_data_type(saliency_map: np.ndarray) -> np.ndarray:
        if saliency_map.dtype != np.uint8:
            saliency_map = saliency_map.astype(np.uint8)
        return saliency_map

    def _select_target_saliency_maps(
        self,
        saliency_map: Dict[Union[int, str], np.ndarray],
        target_explain_group: TargetExplainGroup,
        raw_predictions: ClassificationResult = None,
        explain_targets: List[int] = None
    ) -> Dict[Union[int, str], np.ndarray]:
        # For classification
        if target_explain_group == TargetExplainGroup.IMAGE:
            assert self.get_layout(saliency_map) == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY
            return saliency_map
        elif target_explain_group == TargetExplainGroup.ALL_CLASSES:
            assert self.get_layout(saliency_map) == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
            return saliency_map
        elif target_explain_group in SELECTED_CLASSES:
            assert self.get_layout(saliency_map) == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
            if target_explain_group == TargetExplainGroup.PREDICTED_CLASSES:
                assert raw_predictions is not None, (
                    f"Raw model predictions has to be provided " f"for {target_explain_group}."
                )
                assert raw_predictions.top_labels, (
                    "TargetExplainGroup.PREDICTED_CLASSES requires predictions "
                    "to be available, but currently model has no predictions. "
                    "Try to use different input data, confidence threshold"
                    " or retrain the model."
                )
                assert explain_targets is None, (
                    f"Explain targets do NOT have to be provided for "
                    f"{target_explain_group}. Model prediction is used "
                    f"to retrieve explain targets."
                )
                labels = set([top_prediction[0] for top_prediction in raw_predictions.top_labels])
            else:
                assert (
                    explain_targets is not None
                ), f"Explain targets has to be provided for {target_explain_group}."
                assert (
                    all(0 <= target <= len(saliency_map[0]) - 1 for target in explain_targets)
                ), f"For class-wise targets, all explain targets has to be in range 0..{len(saliency_map[0]) - 1}"
                labels = set(explain_targets)

            saliency_map_predicted_classes = {i: saliency_map[i] for i in labels}
            return saliency_map_predicted_classes
        else:
            raise ValueError(
                f"Target explain group {target_explain_group} is not supported for classification."
            )
        # TODO: implement for detection, probably in a separate class

    @staticmethod
    def get_layout(saliency_map: Dict[int, np.ndarray]) -> SaliencyMapLayout:
        """Estimate and return SaliencyMapLayout. Requires raw saliency map."""
        if "per_image_map" in saliency_map:
            return SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY
        else:
            return SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY

    def save(self, dir_path: str, name: Optional[str] = None) -> None:
        """Dumps saliency map."""
        # TODO: add unit test
        os.makedirs(dir_path, exist_ok=True)
        save_name = f"{name}_" if name else ""
        for idx, map_to_save in self._saliency_map.items():
            target_name = idx if idx == "per_image_map" else self._labels[idx]
            cv2.imwrite(os.path.join(dir_path, f"{save_name}map_{target_name}.jpg"), img=map_to_save)

    @staticmethod
    def reorder_hierarch_sal_map(saliency_map: Dict[int, np.array], hierarchical_info: Dict, labels: List):
        """Reorder saliency maps from hierarchical model output to be adjusted to label-schema labels order."""
        hierarchical_idx = []
        for head_idx in range(hierarchical_info["num_multiclass_heads"]):
            logits_begin, logits_end = hierarchical_info["head_idx_to_logits_range"][str(head_idx)]
            for logit in range(0, logits_end - logits_begin):
                label_str = hierarchical_info["all_groups"][head_idx][logit]
                label_idx = hierarchical_info["label_to_idx"][label_str]
                hierarchical_idx.append(label_idx)

        if hierarchical_info["num_multilabel_classes"]:
            logits_begin = hierarchical_info["num_single_label_classes"]
            logits_end = len(labels)
            for logit_idx, logit in enumerate(range(0, logits_end - logits_begin)):
                label_str_idx = hierarchical_info["num_multiclass_heads"] + logit_idx
                label_str = hierarchical_info["all_groups"][label_str_idx][0]
                label_idx = hierarchical_info["label_to_idx"][label_str]
                hierarchical_idx.append(label_idx)

        reordered_map = {}
        for i, h_idx in enumerate(hierarchical_idx):
            reordered_map[h_idx] = saliency_map[i]
        return dict(sorted(reordered_map.items()))

class PostProcessor:
    """
    PostProcessor implements post-processing for the saliency map.

    :param saliency_map: Input raw saliency map(s).
    :type saliency_map: ExplainResult
    :param data: Input data.
    :type data: ExplainResult
    :param post_processing_parameters: Parameters that define post-processing.
    :type post_processing_parameters: PostProcessParameters
    """

    # TODO: extract public staticmethod methods to operate at pure numpy array level

    def __init__(
        self,
        saliency_map: ExplainResult,
        data: np.ndarray = None,
        post_processing_parameters: PostProcessParameters = PostProcessParameters(),
    ):
        self._saliency_map = saliency_map
        self._data = data

        self._normalize = post_processing_parameters.normalize
        self._resize = post_processing_parameters.resize
        self._colormap = post_processing_parameters.colormap
        self._overlay = post_processing_parameters.overlay
        self._overlay_weight = post_processing_parameters.overlay_weight

    def postprocess(self) -> ExplainResult:
        """
        Saliency map postprocess method.
        Returns ExplainResult object with processed saliency map, that can have one of SaliencyMapLayout layouts.
        """
        saliency_map = self._saliency_map
        # convert to numpy array to use vectorized normalization and speed up lots of classes scenario:
        self._saliency_map.map = np.array([map for _, map in saliency_map.map.items()])

        if self._normalize:
            self._apply_normalization()

        if self._overlay:
            if self._data is None:
                raise ValueError("Input data has to be provided for overlay.")
            self._apply_resize()
            self._apply_colormap()
            self._apply_overlay()
        else:
            if self._resize:
                if self._data is None:
                    # TODO: add explicit target_size as an option
                    raise ValueError("Input data has to be provided for resize (for target size estimation).")
                self._apply_resize()
            if self._colormap:
                self._apply_colormap()
        self.convert_sal_map_to_dict()
        return self._saliency_map

    def _apply_normalization(self) -> None:
        """Normalize saliency maps to [0, 255] range."""
        layout = self._saliency_map.layout
        assert layout in GRAY_LAYOUTS, (
            f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
            f"but got {layout}."
        )
        saliency_map = self._saliency_map.map
        n, h, w = saliency_map.shape
        saliency_map = saliency_map.reshape((n, h*w))
        saliency_map.astype(np.float32)

        min_values, max_values = self._get_min_max(saliency_map)
        saliency_map = 255 * (saliency_map - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        saliency_map = saliency_map.reshape(n, h, w)
        saliency_map = saliency_map.astype(np.uint8)

        self._saliency_map.map = saliency_map

    @staticmethod
    def _get_min_max(saliency_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_values = np.min(saliency_map, axis=-1)
        max_values = np.max(saliency_map, axis=-1)
        return min_values, max_values

    def _apply_resize(self) -> None:
        """Resizes saliency map to the original size of input data."""
        # TODO: support resize of colormapped images.
        # TODO: support resize to custom size.
        layout = self._saliency_map.layout
        assert layout in GRAY_LAYOUTS, (
            f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
            f"but got {layout}."
        )
        saliency_map = self._saliency_map.map
        x = saliency_map.transpose((1, 2, 0))
        x = cv2.resize(x, self._data.shape[:2][::-1])
        if x.ndim == 2:
            saliency_map = np.expand_dims(x, axis=0)
        else:
            saliency_map = x.transpose((2, 0, 1))
        self._saliency_map.map = saliency_map

    def _apply_colormap(self) -> None:
        """Applies cv2.applyColorMap to the saliency map."""
        #  TODO: support different (custom?) colormaps.
        assert self._saliency_map.map.dtype == np.uint8, (
            "Colormap requires saliency map to has uint8 dtype. " 
            "Enable 'normalize' flag for PostProcessor."
        )
        layout = self._saliency_map.layout
        assert layout in GRAY_LAYOUTS, (
            f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
            f"but got {layout}."
        )

        color_mapped_saliency_map = []
        for class_map in self._saliency_map.map:
            color_mapped_saliency_map.append(cv2.applyColorMap(class_map, cv2.COLORMAP_JET))
        color_mapped_saliency_map = np.array(color_mapped_saliency_map)

        if layout == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY:
            self._saliency_map.layout = SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR
        if layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            self._saliency_map.layout = SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR
        self._saliency_map.map = color_mapped_saliency_map

    def _apply_overlay(self) -> None:
        """Applies overlay of the saliency map with the original image."""
        assert (
            self._saliency_map.layout in COLOR_MAPPED_LAYOUTS
        ), "Color mapped saliency map are expected for overlay."

        x = self._saliency_map.map
        x = self._data * self._overlay_weight + x * (1 - self._overlay_weight)
        x[x > 255] = 255
        self._saliency_map.map = x.astype(np.uint8)

    def convert_sal_map_to_dict(self) -> None:
        saliency_map = self._saliency_map.map
        if isinstance(saliency_map, np.ndarray):
            if self._saliency_map.layout in ONE_MAP_LAYOUTS:
                dict_sal_map = {"per_image_map": saliency_map[0]}
                self._saliency_map.map = dict_sal_map
            elif self._saliency_map.layout in MULTIPLE_MAP_LAYOUTS:
                dict_sal_map = {}
                for idx, class_sal in enumerate(saliency_map):
                    dict_sal_map[idx] = class_sal
            self._saliency_map.map = dict_sal_map
