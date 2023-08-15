import os
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from openvino.model_api.models import ClassificationResult


class TargetExplainGroup(Enum):
    """
    Describes target explain groups.

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
    Describes saliency map layout.

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
    TODO: Separate for task type, e.g. create ExplainResult <- ExplainResultClassification, etc.
    """

    def __init__(
        self,
        raw_result: ClassificationResult,
        target_explain_group: TargetExplainGroup,
        explain_targets: Optional[List[int]] = None,
        labels: List[str] = None,
    ):
        saliency_map = self._get_saliency_map_from_predictions(raw_result)
        self._saliency_map = self._select_target_saliency_maps(
            saliency_map, target_explain_group, raw_result, explain_targets
        )
        self._layout = self.get_layout(self._saliency_map)
        self._labels = labels

    @property
    def map(self):
        return self._saliency_map

    @map.setter
    def map(self, saliency_map):
        saliency_map = self._check_data_type(saliency_map)
        self._saliency_map = saliency_map

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
        return raw_saliency_map

    @staticmethod
    def _check_data_type(saliency_map: np.ndarray) -> np.ndarray:
        if saliency_map.dtype != np.uint8:
            saliency_map = saliency_map.astype(np.uint8)
        return saliency_map

    def _select_target_saliency_maps(
        self, saliency_map, target_explain_group, raw_predictions=None, explain_targets=None
    ) -> np.ndarray:
        # For classification
        if target_explain_group == TargetExplainGroup.IMAGE:
            assert self.get_layout(saliency_map) == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY
            return saliency_map
        elif target_explain_group == TargetExplainGroup.ALL_CLASSES:
            assert self.get_layout(saliency_map) == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
            return saliency_map
        elif target_explain_group in SELECTED_CLASSES:
            # TODO: keep track of which maps are selected (e.g. for which classes)
            assert self.get_layout(saliency_map) == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
            if target_explain_group == TargetExplainGroup.PREDICTED_CLASSES:
                assert raw_predictions is not None, (
                    f"Raw model predictions has to be provided " f"for {target_explain_group}."
                )
                assert len(raw_predictions.top_labels) > 0, (
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
                # TODO: support mlc and h-label
                labels = set([top_prediction[0] for top_prediction in raw_predictions.top_labels])
            else:
                assert explain_targets is not None, f"Explain targets has to be provided for {target_explain_group}."
                labels = set(explain_targets)
            saliency_map_predicted_classes = []
            x = saliency_map[0]
            for i in range(len(x)):
                if i in labels:
                    saliency_map_predicted_classes.append(x[i])
            if len(saliency_map_predicted_classes) == 1:
                x = saliency_map_predicted_classes[0]
            else:
                x = np.array(saliency_map_predicted_classes)
            return x[np.newaxis, ...]
        else:
            raise ValueError(f"Target explain group {target_explain_group} is not supported for classification.")
        # TODO: implement for detection, probably in a separate class

    @staticmethod
    def get_layout(saliency_map):
        """Estimate and return SaliencyMapLayout. Requires raw saliency map."""
        if saliency_map.ndim == 3:
            return SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY
        elif saliency_map.ndim == 4:
            return SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
        else:
            raise ValueError(
                f"Raw saliency map has to be three or four dimensional tensor, " f"but got {saliency_map.ndim}."
            )

    def save(self, dir_path, name: Optional[str] = None) -> None:
        """Dumps saliency map."""
        # TODO: use labels instead of map_id for classification
        # TODO: add unit test
        os.makedirs(dir_path, exist_ok=True)
        if self._layout in MULTIPLE_MAP_LAYOUTS:
            batch_size, map_nums, *_ = self._saliency_map.shape
            for i in range(batch_size):
                for map_id in range(map_nums):
                    map_to_save = self._saliency_map[i, map_id]
                    save_name = name if name else i
                    cv2.imwrite(os.path.join(dir_path, f"{save_name}_map{map_id}.jpg"), img=map_to_save)
        if self._layout in ONE_MAP_LAYOUTS:
            batch, *_ = self._saliency_map.shape
            for i in range(batch):
                map_to_save = self._saliency_map[i]
                save_name = name if name else i
                cv2.imwrite(os.path.join(dir_path, f"{save_name}.jpg"), img=map_to_save)


class PostProcessor:
    """PostProcessor implements post-processing for the saliency map.

    Args:
        saliency_map: Input raw saliency map(s).
        data: Input data.
        normalize: If True, normalize saliency map into [0, 255] range (filling the whole range).
            By default, normalization to [0, 255] range is embedded into the IR model.
            Therefore, normalize=False here by default.
        resize: If True, resize saliency map to the input image size.
        colormap: If True, apply colormap to the grayscale saliency map.
        overlay: If True, generate overlay of the saliency map over the input image.
        overlay_weight: Weight of the saliency map when overlaying the input data with the saliency map.
    """

    def __init__(
        self,
        saliency_map: ExplainResult,
        data: np.ndarray = None,
        normalize: bool = False,
        resize: bool = False,
        colormap: bool = False,
        overlay: bool = False,
        overlay_weight: float = 0.5,
    ):
        self._saliency_map = saliency_map
        self._data = data
        self._normalize = normalize
        self._resize = resize
        self._colormap = colormap
        self._overlay = overlay
        self._overlay_weight = overlay_weight

    def postprocess(self) -> ExplainResult:
        """Saliency map postprocess method.

        Return:
            saliency_map: ExplainResult object with processed saliency map, that can have the following layout:
                - B, H, W - single map per image
                - B, H, W, C - single map per image, colormapped
                - B, N, H, W - multiple maps per image, e.g. per-class maps can potentially lead to this
                - B, N, H, W, C - multiple maps per image, colormapped
        """
        if self._normalize:
            self.apply_normalization()

        if self._overlay:
            if self._data is None:
                raise ValueError("Input data has to be provided for overlay.")
            self.apply_resize()
            self.apply_colormap()
            self.apply_overlay()
        else:
            if self._resize:
                if self._data is None:
                    # TODO: add explicit target_size as an option
                    raise ValueError("Input data has to be provided for resize (for target size estimation).")
                self.apply_resize()
            if self._colormap:
                self.apply_colormap()
        return self._saliency_map

    def apply_normalization(self) -> None:
        """Normalize saliency maps to [0, 255] range."""
        saliency_map = self._saliency_map.map.astype(np.float32)
        layout = self._saliency_map.layout
        if layout == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY:
            batch_size, h, w = saliency_map.shape
            saliency_map = saliency_map.reshape((batch_size, h * w))
            min_values, max_values = self._get_min_max(saliency_map)
            saliency_map = 255 * (saliency_map - min_values) / (max_values - min_values + 1e-12)
            saliency_map = saliency_map.reshape((batch_size, h, w))
        elif layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            batch_size, n, h, w = saliency_map.shape
            saliency_map = saliency_map.reshape((batch_size, n, h * w))
            min_values, max_values = self._get_min_max(saliency_map)
            saliency_map = 255 * (saliency_map - min_values[:, :, None]) / (max_values - min_values + 1e-12)[:, :, None]
            saliency_map = saliency_map.reshape((batch_size, n, h, w))
        else:
            raise RuntimeError(
                f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
                f"but got {layout}."
            )
        saliency_map = saliency_map.astype(np.uint8)
        self._saliency_map.map = saliency_map

    @staticmethod
    def _get_min_max(saliency_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_values = np.min(saliency_map, axis=-1)
        max_values = np.max(saliency_map, axis=-1)
        return min_values, max_values

    def apply_resize(self) -> None:
        """
        Resizes saliency map to the original size of input data.
        TODO: support resize of colormapped images.
        TODO: support resize to custom size.
        """
        if self._saliency_map.layout == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY:
            x = self._saliency_map.map[0]
            x = cv2.resize(x, self._data.shape[:2][::-1])
            self._saliency_map.map = x[np.newaxis, ...]
        elif self._saliency_map.layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            x = self._saliency_map.map[0]
            x = x.transpose((1, 2, 0))
            x = cv2.resize(x, self._data.shape[:2][::-1])
            x = x.transpose((2, 0, 1))
            self._saliency_map.map = x[np.newaxis, ...]
        else:
            raise RuntimeError(
                f"Saliency map layout has to be in {GRAY_LAYOUTS}, " f"but got {self._saliency_map.layout}."
            )

    def apply_colormap(self) -> None:
        """Applies cv2.applyColorMap to the saliency map
        TODO: support different (custom?) colormaps."""
        if self._saliency_map.layout == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY:
            x = self._saliency_map.map[0]
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            self._saliency_map.map = x[np.newaxis, ...]
            self._saliency_map.layout = SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR
        elif self._saliency_map.layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            x = self._saliency_map.map[0]
            num_maps = x.shape[0]
            color_mapped_saliency_map = []
            # TODO: is it possible to vectorize colormapping?
            for i in range(num_maps):
                color_mapped_saliency_map.append(cv2.applyColorMap(x[i], cv2.COLORMAP_JET))
            x = np.array(color_mapped_saliency_map)
            self._saliency_map.map = x[np.newaxis, ...]
            self._saliency_map.layout = SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR
        else:
            raise RuntimeError(
                f"Saliency map layout has to be in {GRAY_LAYOUTS}, " f"but got {self._saliency_map.layout}."
            )

    def apply_overlay(self) -> None:
        """Applies overlay of the saliency map with the original image."""
        assert self._saliency_map.layout in COLOR_MAPPED_LAYOUTS, "Color mapped saliency map are expected for overlay."
        x = self._saliency_map.map[0]
        x = self._data * self._overlay_weight + x * (1 - self._overlay_weight)
        x[x > 255] = 255
        x = x.astype(np.uint8)
        self._saliency_map.map = x[np.newaxis, ...]
