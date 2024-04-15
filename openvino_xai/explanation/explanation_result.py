# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from openvino_xai.common.utils import logger
from openvino_xai.explanation.explanation_parameters import TargetExplainGroup, SaliencyMapLayout
from openvino_xai.explanation.utils import select_target_indices


class ExplanationResult:
    """
    ExplanationResult selects target saliency maps, holds it and its layout.

    :param saliency_map: Raw saliency map.
    :param target_explain_group: Defines targets to explain: all, only predictions, custom list, per-image.
    :type target_explain_group: TargetExplainGroup
    :param target_explain_indices: List of custom targets, optional.
    :type target_explain_indices: Optional[List[int]]
    :param target_explain_names: List of all target_explain_names.
    :type target_explain_names: List[str]
    """

    def __init__(
        self,
        saliency_map: np.ndarray,
        target_explain_group: TargetExplainGroup,
        target_explain_indices: Optional[List[int]] = None,
        target_explain_names: Optional[List[str]] = None,
    ):
        self._check_saliency_map(saliency_map)
        self.saliency_map = self._format_sal_map_as_dict(saliency_map)

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

        if self.target_explain_group == TargetExplainGroup.CUSTOM:
            self.saliency_map = self._select_target_saliency_maps(target_explain_indices)

        self.target_explain_names = target_explain_names

    @property
    def sal_map_shape(self):
        idx = next(iter(self.saliency_map))
        sal_map_shape = self.saliency_map[idx].shape
        return sal_map_shape

    @staticmethod
    def _check_saliency_map(saliency_map: np.ndarray):
        if saliency_map is None:
            raise RuntimeError("Saliency map is None.")
        if not isinstance(saliency_map, np.ndarray):
            raise ValueError(f"Raw saliency_map has to be np.ndarray, but got {type(saliency_map)}.")
        if saliency_map.size == 0:
            raise RuntimeError("Saliency map is zero size array.")
        if saliency_map.shape[0] > 1:
            raise RuntimeError("Batch size for saliency maps should be 1.")

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
        self, target_explain_indices: Optional[List[int]] = None
    ) -> Dict[Union[int, str], np.ndarray]:
        assert self.layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
        explain_target_indexes = select_target_indices(
            self.target_explain_group,
            target_explain_indices,
            len(self.saliency_map),
        )
        saliency_maps_selected = {i: self.saliency_map[i] for i in explain_target_indexes}
        return saliency_maps_selected

    def save(self, dir_path: Union[Path, str], name: Optional[str] = None) -> None:
        """Dumps saliency map."""
        # TODO: add unit test
        os.makedirs(dir_path, exist_ok=True)
        save_name = f"{name}_" if name else ""
        for i, (cls_idx, map_to_save) in enumerate(self.saliency_map.items()):
            if cls_idx == "per_image_map":
                target_name = "per_image_map"
            else:
                if self.target_explain_names:
                    target_name = self.target_explain_names[cls_idx]
                else:
                    target_name = cls_idx
            cv2.imwrite(os.path.join(dir_path, f"{save_name}target_{target_name}.jpg"), img=map_to_save)
