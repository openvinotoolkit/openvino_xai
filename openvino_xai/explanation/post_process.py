# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List

import cv2
import numpy as np

from openvino_xai.explanation.explanation_parameters import PostProcessParameters, SaliencyMapLayout, GRAY_LAYOUTS, \
    COLOR_MAPPED_LAYOUTS, MULTIPLE_MAP_LAYOUTS, ONE_MAP_LAYOUTS
from openvino_xai.explanation.explanation_result import ExplanationResult


class PostProcessor:
    """
    PostProcessor implements post-processing for the explanation result.

    :param explanation: Explanation result object.
    :type explanation: ExplanationResult
    :param data: Input data.
    :type data: ExplanationResult
    :param post_processing_parameters: Parameters that define post-processing.
    :type post_processing_parameters: PostProcessParameters
    """

    # TODO: extract public staticmethod methods to operate at pure numpy array level
    # TODO: add unit tests with reference values for each of methods

    def __init__(
        self,
        explanation: ExplanationResult,
        data: np.ndarray = None,
        post_processing_parameters: PostProcessParameters = PostProcessParameters(),
    ):
        self._explanation = explanation
        self._data = data

        self._normalize = post_processing_parameters.normalize
        self._resize = post_processing_parameters.resize
        self._colormap = post_processing_parameters.colormap
        self._overlay = post_processing_parameters.overlay
        self._overlay_weight = post_processing_parameters.overlay_weight

    def postprocess(self) -> ExplanationResult:
        """
        Saliency map postprocess method.
        Returns ExplainResult object with processed saliency map, that can have one of SaliencyMapLayout layouts.
        """
        saliency_map = self._explanation
        class_idx_to_return = list(saliency_map.saliency_map.keys())
        # convert to numpy array to use vectorized normalization and speed up lots of classes scenario:
        self._explanation.saliency_map = np.array(list(saliency_map.saliency_map.values()))

        if self._normalize and not self._resize and not self._overlay:
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
        self.convert_sal_map_to_dict(class_idx_to_return)
        return self._explanation

    def _apply_normalization(self) -> None:
        """Normalize saliency maps to [0, 255] range."""
        layout = self._explanation.layout
        assert layout in GRAY_LAYOUTS, (
            f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
            f"but got {layout}."
        )
        saliency_map = self._explanation.saliency_map
        n, h, w = saliency_map.shape
        saliency_map = saliency_map.reshape((n, h*w))
        saliency_map = saliency_map.astype(np.float32)

        min_values, max_values = self._get_min_max(saliency_map)
        saliency_map = 255 * (saliency_map - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        saliency_map = saliency_map.reshape(n, h, w)
        saliency_map = saliency_map.astype(np.uint8)

        self._explanation.saliency_map = saliency_map

    @staticmethod
    def _get_min_max(saliency_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_values = np.min(saliency_map, axis=-1)
        max_values = np.max(saliency_map, axis=-1)
        return min_values, max_values

    def _apply_resize(self) -> None:
        """Resizes saliency map to the original size of input data."""
        # TODO: support resize of colormapped images.
        # TODO: support resize to custom size.
        layout = self._explanation.layout
        assert layout in GRAY_LAYOUTS, (
            f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
            f"but got {layout}."
        )
        saliency_map = self._explanation.saliency_map
        x = saliency_map.transpose((1, 2, 0))
        x = cv2.resize(x, self._data.shape[:2][::-1])
        if x.ndim == 2:
            saliency_map = np.expand_dims(x, axis=0)
        else:
            saliency_map = x.transpose((2, 0, 1))
        self._explanation.saliency_map = saliency_map

        # Normalization has to be applied after resize to keep map in range 0..255
        self._apply_normalization()

    def _apply_colormap(self) -> None:
        """Applies cv2.applyColorMap to the saliency map."""
        #  TODO: support different (custom?) colormaps.
        assert self._explanation.saliency_map.dtype == np.uint8, (
            "Colormap requires saliency map to has uint8 dtype. "
            "Enable 'normalize' flag for PostProcessor."
        )
        layout = self._explanation.layout
        assert layout in GRAY_LAYOUTS, (
            f"Saliency map to normalize has to be grayscale. Layout must be in {GRAY_LAYOUTS}, "
            f"but got {layout}."
        )

        color_mapped_saliency_map = []
        for class_map in self._explanation.saliency_map:
            color_mapped_saliency_map.append(cv2.applyColorMap(class_map, cv2.COLORMAP_JET))
        color_mapped_saliency_map = np.array(color_mapped_saliency_map)

        if layout == SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY:
            self._explanation.layout = SaliencyMapLayout.ONE_MAP_PER_IMAGE_COLOR
        if layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            self._explanation.layout = SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_COLOR
        self._explanation.saliency_map = color_mapped_saliency_map

    def _apply_overlay(self) -> None:
        """Applies overlay of the saliency map with the original image."""
        assert (
                self._explanation.layout in COLOR_MAPPED_LAYOUTS
        ), "Color mapped saliency map are expected for overlay."

        x = self._explanation.saliency_map
        x = self._data * self._overlay_weight + x * (1 - self._overlay_weight)
        x[x > 255] = 255
        self._explanation.saliency_map = x.astype(np.uint8)

    def convert_sal_map_to_dict(self, class_idx: List) -> None:
        saliency_map = self._explanation.saliency_map
        if isinstance(saliency_map, np.ndarray):
            if self._explanation.layout in ONE_MAP_LAYOUTS:
                dict_sal_map = {"per_image_map": saliency_map[0]}
                self._explanation.saliency_map = dict_sal_map
            elif self._explanation.layout in MULTIPLE_MAP_LAYOUTS:
                dict_sal_map = {}
                for idx, class_sal in zip(class_idx, saliency_map):
                    dict_sal_map[idx] = class_sal
            else:
                raise ValueError
            self._explanation.saliency_map = dict_sal_map
