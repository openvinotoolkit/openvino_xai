# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from openvino_xai.common.parameters import Task
from openvino_xai.common.utils import logger
from openvino_xai.explainer.utils import (
    convert_targets_to_numpy,
    explains_all,
    get_target_indices,
)
from openvino_xai.methods.base import Prediction


class Explanation:
    """
    Explanation selects target saliency maps, holds it and its layout.

    :param saliency_map: Raw saliency map, as a numpy array or as a dict.
    :type saliency_map: np.ndarray | Dict[int | str, np.ndarray]
    :param targets: List of custom labels to explain, optional. Can be list of integer indices (int),
        or list of names (str) from label_names.
    :type targets: np.ndarray | List[int | str] | int | str
    :param task: Type of the task: CLASSIFICATION or DETECTION.
    :type task: Task
    :param label_names: List of all label names.
    :type label_names: List[str] | None
    :param predictions: Per-target model prediction (available only for black-box methods).
    :type predictions: Dict[int, Prediction] | None
    """

    def __init__(
        self,
        saliency_map: np.ndarray | Dict[int | str, np.ndarray],
        targets: np.ndarray | List[int | str] | int | str,
        task: Task,
        label_names: List[str] | None = None,
        predictions: Dict[int, Prediction] | None = None,
    ):
        targets = convert_targets_to_numpy(targets)

        if isinstance(saliency_map, np.ndarray):
            self._check_saliency_map(saliency_map)
            self._saliency_map = self._format_sal_map_as_dict(saliency_map)
            self.total_num_targets = len(self._saliency_map)
        elif isinstance(saliency_map, dict):
            self._saliency_map = saliency_map
            self.total_num_targets = None
        else:
            raise ValueError(f"Expect saliency_map to be np.ndarray or dict, but got{type(saliency_map)}.")

        if "per_image_map" in self._saliency_map:
            self.layout = Layout.ONE_MAP_PER_IMAGE_GRAY
        else:
            self.layout = Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY

        if not explains_all(targets) and not self.layout == Layout.ONE_MAP_PER_IMAGE_GRAY:
            label_names_ = None if task == Task.DETECTION else label_names
            self._saliency_map = self._select_target_saliency_maps(targets, label_names_)

        self.task = task
        self.label_names = label_names
        self.predictions = predictions

    @property
    def saliency_map(self) -> Dict[int | str, np.ndarray]:
        """Saliency map as a dict {target_id: np.ndarray}."""
        return self._saliency_map

    @saliency_map.setter
    def saliency_map(self, saliency_map: Dict[int | str, np.ndarray]):
        self._saliency_map = saliency_map

    @property
    def shape(self):
        """Shape of the saliency map."""
        idx = next(iter(self._saliency_map))
        shape = self._saliency_map[idx].shape
        return shape

    @property
    def targets(self):
        """Explained targets."""
        return list(self._saliency_map.keys())

    @staticmethod
    def _check_saliency_map(saliency_map: np.ndarray):
        if saliency_map is None:
            raise RuntimeError("Saliency map is None.")
        if saliency_map.size == 0:
            raise RuntimeError("Saliency map is zero size array.")
        if saliency_map.shape[0] > 1:
            raise RuntimeError("Batch size for saliency maps should be 1.")

    @staticmethod
    def _format_sal_map_as_dict(raw_saliency_map: np.ndarray) -> Dict[int | str, np.ndarray]:
        """Returns dict with saliency maps in format {target_id: class_saliency_map}."""
        dict_sal_map: Dict[int | str, np.ndarray]
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
        self,
        targets: np.ndarray | List[int | str],
        label_names: List[str] | None = None,
    ) -> Dict[int | str, np.ndarray]:
        assert self.layout == Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY
        target_indices = self._select_target_indices(
            targets=targets,
            label_names=label_names,
        )
        saliency_maps_selected = {i: self._saliency_map[i] for i in target_indices}
        return saliency_maps_selected

    def _select_target_indices(
        self,
        targets: np.ndarray | List[int | str],
        label_names: List[str] | None = None,
    ) -> List[int] | np.ndarray:
        target_indices = get_target_indices(targets, label_names)
        if self.total_num_targets is not None:
            if not all(0 <= target_index <= (self.total_num_targets - 1) for target_index in target_indices):
                raise ValueError(f"All targets indices have to be in range 0..{self.total_num_targets - 1}.")
        else:
            if not all(target_index in self.saliency_map for target_index in target_indices):
                raise ValueError("Provided targer index {targer_index} is not available among saliency maps.")
        return target_indices

    def save(
        self,
        dir_path: Path | str,
        prefix: str = "",
        postfix: str = "",
        confidence_scores: Dict[int, float] | None = None,
    ) -> None:
        """
        Dumps saliency map images to the specified directory.

        Allows flexibly name the files with the prefix and postfix.
        {prefix} + target_id + {postfix}.jpg

        Also allows to add confidence scores to the file names.
        {prefix} + target_id + {postfix} + confidence.jpg

        save(output_dir) -> aeroplane.jpg
        save(output_dir, prefix="image_name_target_") -> image_name_target_aeroplane.jpg
        save(output_dir, postfix="_class_map") -> aeroplane_class_map.jpg
        save(
            output_dir, prefix="image_name_", postfix="_conf_", confidence_scores=scores
        ) -> image_name_aeroplane_conf_0.85.jpg

        Parameters:
            :param dir_path: The directory path where the saliency maps will be saved.
            :type dir_path: Path | str
            :param prefix: Optional prefix for the saliency map names. Default is an empty string.
            :type prefix: str
            :param postfix: Optional postfix for the saliency map names. Default is an empty string.
            :type postfix: str
            :param confidence_scores: Dict with confidence scores for each class index. Default is None.
            :type confidence_scores: Dict[int, float] | None
        """

        os.makedirs(dir_path, exist_ok=True)

        template = f"{prefix}{{target_name}}{postfix}{{conf_score}}.jpg"
        for target_idx, map_to_save in self._saliency_map.items():
            conf_score = ""
            map_to_save = cv2.cvtColor(map_to_save, code=cv2.COLOR_RGB2BGR)
            if isinstance(target_idx, str):
                target_name = "activation_map"
            elif self.label_names and isinstance(target_idx, (int, np.int64)) and self.task != Task.DETECTION:
                target_name = self.label_names[target_idx]
            else:
                target_name = str(target_idx)

            if confidence_scores and target_idx in confidence_scores:
                conf_score = f"{confidence_scores[int(target_idx)]:.2f}"

            image_name = template.format(target_name=target_name, conf_score=conf_score)
            cv2.imwrite(os.path.join(dir_path, image_name), img=map_to_save)

    def plot(
        self,
        targets: np.ndarray | List[int | str] | None = None,
        backend: str = "matplotlib",
        max_num_plots: int = 24,
        num_columns: int = 4,
    ) -> None:
        """
        Plots saliency maps using the specified backend.

        This function plots available saliency maps using the specified backend. Targets to plot
        can be specified by passing a list of target class indices or names. If a provided class is
        not available among the saliency maps, it is omitted.

        Args:
            targets (np.ndarray | List[int | str] | None): A list or array of target class indices or names to plot.
                By default, it's None, and all available saliency maps are plotted.
            backend (str): The plotting backend to use. Can be either 'matplotlib' (recommended for Jupyter)
                or 'cv' (recommended for Python scripts). Default is 'matplotlib'.
            max_num_plots (int): Max number of images to plot. Default is 24 to avoid memory issues.
            num_columns (int): Number of columns in the saliency maps visualization grid for the matplotlib backend.
        """

        if targets is None or explains_all(targets):
            checked_targets = self.targets
        else:
            target_indices = get_target_indices(targets, self.label_names)
            checked_targets = []
            for target_index in target_indices:
                if target_index in self.saliency_map:
                    checked_targets.append(target_index)
                else:
                    logger.info(f"Provided class index {target_index} is not available among saliency maps.")

        if len(checked_targets) > max_num_plots:
            logger.warning(
                f"Decrease the number of plotted saliency maps from {len(checked_targets)} to {max_num_plots}"
                " to avoid the memory issue. To avoid this, increase the 'max_num_plots' argument."
            )
            checked_targets = checked_targets[:max_num_plots]

        if backend == "matplotlib":
            self._plot_matplotlib(checked_targets, num_columns)
        elif backend == "cv":
            self._plot_cv(checked_targets)
        else:
            raise ValueError(f"Unknown backend {backend}. Use 'matplotlib' or 'cv'.")

    def _plot_matplotlib(self, checked_targets: list[int | str], num_cols: int) -> None:
        """Plots saliency maps using matplotlib."""
        num_rows = int(np.ceil(len(checked_targets) / num_cols))
        _, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 6 * num_rows))
        axes = axes.flatten()

        for i, target_index in enumerate(checked_targets):
            if self.label_names and isinstance(target_index, np.int64):
                label_name = f"{self.label_names[target_index]} ({target_index})"
            else:
                label_name = str(target_index)

            map_to_plot = self.saliency_map[target_index]

            if map_to_plot.ndim == 3:
                axes[i].imshow(map_to_plot)
            elif map_to_plot.ndim == 2:
                axes[i].imshow(map_to_plot, cmap="gray")
            else:
                raise ValueError(f"Saliency map expected to be 3 or 2-dimensional, but got {map_to_plot.ndim}.")
            axes[i].axis("off")  # Hide the axis
            axes[i].set_title(f"Class {label_name}")

        # Hide remaining axes
        for ax in axes[len(checked_targets) :]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def _plot_cv(self, checked_targets: list[int | str]) -> None:
        """Plots saliency maps using OpenCV."""
        for target_index in checked_targets:
            if self.label_names and isinstance(target_index, np.int64):
                label_name = f"{self.label_names[target_index]} ({target_index})"
            else:
                label_name = str(target_index)

            map_to_plot = self.saliency_map[target_index]
            map_to_plot = cv2.cvtColor(map_to_plot, cv2.COLOR_BGR2RGB)

            cv2.imshow(f"Class {label_name}", map_to_plot)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


class Layout(Enum):
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
    MULTIPLE_MAPS_PER_IMAGE_GRAY = "multiple_maps_per_image_gray"
    MULTIPLE_MAPS_PER_IMAGE_COLOR = "multiple_maps_per_image_color"


GRAY_LAYOUTS = {
    Layout.ONE_MAP_PER_IMAGE_GRAY,
    Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
}
COLOR_MAPPED_LAYOUTS = {
    Layout.ONE_MAP_PER_IMAGE_COLOR,
    Layout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
MULTIPLE_MAP_LAYOUTS = {
    Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
    Layout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
ONE_MAP_LAYOUTS = {
    Layout.ONE_MAP_PER_IMAGE_GRAY,
    Layout.ONE_MAP_PER_IMAGE_COLOR,
}
