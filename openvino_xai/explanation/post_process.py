# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import cv2
import numpy as np

from openvino_xai.common.utils import scale
from openvino_xai.explanation.explanation_parameters import (
    COLOR_MAPPED_LAYOUTS,
    GRAY_LAYOUTS,
    MULTIPLE_MAP_LAYOUTS,
    ONE_MAP_LAYOUTS,
    PostProcessParameters,
    SaliencyMapLayout,
)
from openvino_xai.explanation.explanation_result import ExplanationResult


def resize(saliency_map: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize saliency map."""
    x = saliency_map.transpose((1, 2, 0))
    x = cv2.resize(x, output_size[::-1])
    if x.ndim == 2:
        return np.expand_dims(x, axis=0)
    return x.transpose((2, 0, 1))


def colormap(saliency_map: np.ndarray, colormap_type: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Applies colormap to the saliency map."""
    # Note: inefficient operation. Is there a way to vectorize it?
    color_mapped_saliency_map = []
    for class_map in saliency_map:
        color_mapped_saliency_map.append(cv2.applyColorMap(class_map, colormap_type))
    return np.array(color_mapped_saliency_map)


def overlay(saliency_map: np.ndarray, input_image: np.ndarray, overlay_weight: float = 0.5) -> np.ndarray:
    """Applies overlay of the saliency map with the original image."""
    res = input_image * overlay_weight + saliency_map * (1 - overlay_weight)
    res[res > 255] = 255
    return res


class PostProcessor:
    """
    PostProcessor implements post-processing for the explanation result.

    :param explanation: Explanation result object.
    :type explanation: ExplanationResult
    :param data: Input data.
    :type data: np.ndarray
    :param output_size: Output size used for resize operation.
    :type output_size: Tuple[int, int]
    :param post_processing_parameters: Parameters that define post-processing.
    :type post_processing_parameters: PostProcessParameters
    """

    def __init__(
        self,
        explanation: ExplanationResult,
        data: np.ndarray = None,
        output_size: Tuple[int, int] = None,
        post_processing_parameters: PostProcessParameters | None = None,
    ):
        self._explanation = explanation
        self._saliency_map_np: np.ndarray | None = None
        self._data = data
        self._output_size = output_size

        if post_processing_parameters is None:
            post_processing_parameters = PostProcessParameters(overlay=True)
        self._scale = post_processing_parameters.scale
        self._resize = post_processing_parameters.resize
        self._colormap = post_processing_parameters.colormap
        self._overlay = post_processing_parameters.overlay
        self._overlay_weight = post_processing_parameters.overlay_weight

    @property
    def layout(self) -> SaliencyMapLayout:
        return self._explanation.layout

    @layout.setter
    def layout(self, layout: SaliencyMapLayout):
        self._explanation.layout = layout

    def run(self) -> ExplanationResult:
        """
        Saliency map postprocess method.
        Applies some op ordering logic, depending on PostProcessParameters.
        Returns ExplainResult object with processed saliency map, that can have one of SaliencyMapLayout layouts.
        """
        saliency_map_dict = self._explanation.saliency_map
        class_idx_to_return = list(saliency_map_dict.keys())

        # Convert to numpy array to use vectorized normalization and speed up lots of classes scenario
        self._saliency_map_np = np.array(list(saliency_map_dict.values()))

        if self._scale and not self._resize and not self._overlay:
            self._apply_normalization()

        if self._overlay:
            if self._data is None:
                raise ValueError("Input data has to be provided for overlay.")
            self._apply_resize()
            self._apply_colormap()
            self._apply_overlay()
        else:
            if self._resize:
                if self._data is None and self._output_size is None:
                    raise ValueError(
                        "Input data or output_size has to be provided for resize (for target size estimation)."
                    )
                self._apply_resize()
            if self._colormap:
                self._apply_colormap()

        # Convert back to dict
        self._convert_sal_map_to_dict(class_idx_to_return)
        return self._explanation

    def _apply_normalization(self) -> None:
        if self.layout not in GRAY_LAYOUTS:
            raise ValueError(
                f"Saliency map to scale has to be grayscale. The layout must be in {GRAY_LAYOUTS}, "
                f"but got {self.layout}."
            )
        self._saliency_map_np = scale(self._saliency_map_np)

    def _apply_resize(self) -> None:
        # TODO: support resize of colormapped images.
        if self.layout not in GRAY_LAYOUTS:
            raise ValueError(
                f"Saliency map to resize has to be grayscale. The layout must be in {GRAY_LAYOUTS}, "
                f"but got {self.layout}."
            )
        output_size = self._output_size if self._output_size else self._data.shape[:2]
        self._saliency_map_np = resize(self._saliency_map_np, output_size)

        # Normalization has to be applied after resize to keep map in range 0..255
        self._apply_normalization()

    def _apply_colormap(self) -> None:
        if self._saliency_map_np.dtype != np.uint8:
            raise ValueError(
                "Colormap requires saliency map to has uint8 dtype. Enable 'scale' flag for PostProcessor."
            )
        if self.layout not in GRAY_LAYOUTS:
            raise ValueError(
                f"Saliency map to colormap has to be grayscale. The layout must be in {GRAY_LAYOUTS}, "
                f"but got {self.layout}."
            )
        self._saliency_map_np = colormap(self._saliency_map_np)
        if self.layout == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY:
            self.layout = SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR
        if self.layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            self.layout = SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR

    def _apply_overlay(self) -> None:
        assert self.layout in COLOR_MAPPED_LAYOUTS, "Color mapped saliency map are expected for overlay."
        self._saliency_map_np = overlay(self._saliency_map_np, self._data, self._overlay_weight)

    def _convert_sal_map_to_dict(self, class_idx: List) -> None:
        dict_sal_map: Dict[int | str, np.ndarray] = {}
        if self.layout in ONE_MAP_LAYOUTS:
            dict_sal_map["per_image_map"] = self._saliency_map_np[0]
            self._saliency_map_np = dict_sal_map
        elif self.layout in MULTIPLE_MAP_LAYOUTS:
            for idx, class_sal in zip(class_idx, self._saliency_map_np):
                dict_sal_map[idx] = class_sal
        else:
            raise ValueError
        self._explanation.saliency_map = dict_sal_map
