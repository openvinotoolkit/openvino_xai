# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import cv2
import numpy as np

from openvino_xai.common.utils import format_to_bhwc, infer_size_from_image, scaling
from openvino_xai.explainer.explanation import (
    COLOR_MAPPED_LAYOUTS,
    GRAY_LAYOUTS,
    MULTIPLE_MAP_LAYOUTS,
    ONE_MAP_LAYOUTS,
    Explanation,
    Layout,
)


def resize(saliency_map: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize saliency map."""
    if saliency_map.ndim == 2:
        return cv2.resize(saliency_map, output_size[::-1])

    x = saliency_map.transpose((1, 2, 0))
    # Resizing in batches to prevent memory issue for saliency maps with many
    # (targets=all classes scenario)
    batch_size = 500
    channels = x.shape[-1]
    resized_batches = []
    for start_idx in range(0, channels, batch_size):
        end_idx = min(start_idx + batch_size, channels)
        batch = x[:, :, start_idx:end_idx]
        resized_batch = cv2.resize(batch, output_size[::-1])

        # Ensure resized batch has three dimensions
        if resized_batch.ndim == 2:
            resized_batch = np.expand_dims(resized_batch, axis=2)
        resized_batches.append(resized_batch)

    x = np.concatenate(resized_batches, axis=-1)
    return x.transpose((2, 0, 1))


def colormap(saliency_map: np.ndarray, colormap_type: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Applies colormap to the saliency map."""
    # Note: inefficient operation. Is there a way to vectorize it?
    color_mapped_saliency_map = []
    for class_map in saliency_map:
        colormapped = cv2.applyColorMap(class_map, colormap_type)  # OpenCV: BGR order
        colormapped_rgb = cv2.cvtColor(colormapped, code=cv2.COLOR_BGR2RGB)
        color_mapped_saliency_map.append(colormapped_rgb)
    return np.array(color_mapped_saliency_map)


def overlay(
    saliency_map: np.ndarray, input_image: np.ndarray, overlay_weight: float = 0.5, cast_to_uint8: bool = True
) -> np.ndarray:
    """Applies overlay of the saliency map with the original image."""
    res = input_image * overlay_weight + saliency_map * (1 - overlay_weight)
    res[res > 255] = 255
    if cast_to_uint8:
        return res.astype(np.uint8)
    return res


class Visualizer:
    """
    Visualizer implements post-processing for the saliency map in explanation result.
    """

    def __call__(
        self,
        explanation: Explanation,
        original_input_image: np.ndarray | None = None,
        output_size: Tuple[int, int] = None,
        scaling: bool = False,
        resize: bool = True,
        colormap: bool = True,
        overlay: bool = False,
        overlay_weight: float = 0.5,
    ) -> Explanation:
        return self.visualize(
            explanation,
            original_input_image,
            output_size,
            scaling,
            resize,
            colormap,
            overlay,
            overlay_weight,
        )

    def visualize(
        self,
        explanation: Explanation,
        original_input_image: np.ndarray | None = None,
        output_size: Tuple[int, int] = None,
        scaling: bool = False,
        resize: bool = True,
        colormap: bool = True,
        overlay: bool = False,
        overlay_weight: float = 0.5,
    ) -> Explanation:
        """
        Saliency map postprocess method.
        Applies some op ordering logic, depending on VisualizationParameters.
        Returns ExplainResult object with processed saliency map, that can have one of Layout layouts.

        :param explanation: Explanation result object.
        :type explanation: Explanation
        :param original_input_image: Input original_input_image.
        :type original_input_image: np.ndarray
        :param output_size: Output size used for resize operation.
        :type output_size: Tuple[int, int]
        :parameter scaling: If True, scaling saliency map into [0, 255] range (filling the whole range).
            By default, scaling is embedded into the IR model.
            Therefore, scaling=False here by default.
        :type scaling: bool
        :parameter resize: If True, resize saliency map to the input image size.
        :type resize: bool
        :parameter colormap: If True, apply colormap to the grayscale saliency map.
        :type colormap: bool
        :parameter overlay: If True, generate overlay of the saliency map over the input image.
        :type overlay: bool
        :parameter overlay_weight: Weight of the saliency map when overlaying the input data with the saliency map.
        :type overlay_weight: float
        """
        if original_input_image is not None:
            original_input_image = format_to_bhwc(original_input_image)

        saliency_map_dict = explanation.saliency_map
        class_idx_to_return = list(saliency_map_dict.keys())

        # Convert to numpy array to use vectorized scale (0 ~ 255) operation and speed up lots of classes scenario
        saliency_map_np = np.array(list(saliency_map_dict.values()))

        if scaling and not resize and not overlay:
            saliency_map_np = self._apply_scaling(explanation, saliency_map_np)

        if overlay:
            if original_input_image is None:
                raise ValueError("Input data has to be provided for overlay.")
            saliency_map_np = self._apply_resize(explanation, saliency_map_np, original_input_image, output_size)
            saliency_map_np = self._apply_colormap(explanation, saliency_map_np)
            saliency_map_np = self._apply_overlay(
                explanation, saliency_map_np, original_input_image, output_size, overlay_weight
            )
        else:
            if resize:
                if original_input_image is None and output_size is None:
                    raise ValueError(
                        "Input data or output_size has to be provided for resize (for target size estimation)."
                    )
                saliency_map_np = self._apply_resize(explanation, saliency_map_np, original_input_image, output_size)
            if colormap:
                saliency_map_np = self._apply_colormap(explanation, saliency_map_np)

        # Convert back to dict
        return self._update_explanation_with_processed_sal_map(explanation, saliency_map_np, class_idx_to_return)

    @staticmethod
    def _apply_scaling(explanation: Explanation, saliency_map_np: np.ndarray) -> np.ndarray:
        if explanation.layout not in GRAY_LAYOUTS:
            raise ValueError(
                f"Saliency map to scale has to be grayscale. The layout must be in {GRAY_LAYOUTS}, "
                f"but got {explanation.layout}."
            )
        return scaling(saliency_map_np)

    def _apply_resize(
        self,
        explanation: Explanation,
        saliency_map_np: np.ndarray,
        original_input_image: np.ndarray = None,
        output_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        # TODO: support resize of colormapped images.
        if explanation.layout not in GRAY_LAYOUTS:
            raise ValueError(
                f"Saliency map to resize has to be grayscale. The layout must be in {GRAY_LAYOUTS}, "
                f"but got {explanation.layout}."
            )
        output_size = output_size if output_size else infer_size_from_image(original_input_image)
        saliency_map_np = resize(saliency_map_np, output_size)

        # Scaling has to be applied after resize to keep map in range 0..255
        return self._apply_scaling(explanation, saliency_map_np)

    @staticmethod
    def _apply_colormap(explanation: Explanation, saliency_map_np: np.ndarray) -> np.ndarray:
        if saliency_map_np.dtype != np.uint8:
            raise ValueError("Colormap requires saliency map to has uint8 dtype. Enable 'scaling' flag for Visualizer.")
        if explanation.layout not in GRAY_LAYOUTS:
            raise ValueError(
                f"Saliency map to colormap has to be grayscale. The layout must be in {GRAY_LAYOUTS}, "
                f"but got {explanation.layout}."
            )
        saliency_map_np = colormap(saliency_map_np)
        if explanation.layout == Layout.ONE_MAP_PER_IMAGE_GRAY:
            explanation.layout = Layout.ONE_MAP_PER_IMAGE_COLOR
        if explanation.layout == Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY:
            explanation.layout = Layout.MULTIPLE_MAPS_PER_IMAGE_COLOR
        return saliency_map_np

    @staticmethod
    def _apply_overlay(
        explanation: Explanation,
        saliency_map_np: np.ndarray,
        original_input_image: np.ndarray = None,
        output_size: Tuple[int, int] = None,
        overlay_weight: float = 0.5,
    ) -> np.ndarray:
        if explanation.layout not in COLOR_MAPPED_LAYOUTS:
            raise RuntimeError("Color mapped saliency map are expected for overlay.")
        if output_size:
            original_input_image = cv2.resize(original_input_image[0], output_size[::-1])
            original_input_image = original_input_image[None, ...]
        return overlay(saliency_map_np, original_input_image, overlay_weight)

    @staticmethod
    def _update_explanation_with_processed_sal_map(
        explanation: Explanation,
        saliency_map_np: np.ndarray,
        class_idx: List,
    ) -> Explanation:
        dict_sal_map: Dict[int | str, np.ndarray] = {}
        if explanation.layout in ONE_MAP_LAYOUTS:
            dict_sal_map["per_image_map"] = saliency_map_np[0]
            saliency_map_np = dict_sal_map
        elif explanation.layout in MULTIPLE_MAP_LAYOUTS:
            for idx, class_sal in zip(class_idx, saliency_map_np):
                dict_sal_map[idx] = class_sal
        else:
            raise ValueError
        explanation.saliency_map = dict_sal_map
        return explanation
