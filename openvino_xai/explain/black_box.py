# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List

import cv2
import numpy as np
from tqdm import tqdm

from openvino.model_api.models import ClassificationModel, ClassificationResult

from openvino_xai.explain.base import Explainer
from openvino_xai.parameters import PostProcessParameters
from openvino_xai.saliency_map import ExplainResult, TargetExplainGroup


class BlackBoxExplainer(Explainer):
    """Base class for explainers that consider model as a black-box."""


class RISEExplainer(BlackBoxExplainer):
    """RISEExplainer explains classification models in black-box mode using RISE (https://arxiv.org/abs/1806.07421)."""

    def __init__(self,
                 model: ClassificationModel,
                 num_masks: Optional[int] = 5000,
                 num_cells: Optional[int] = 8,
                 prob: Optional[float] = 0.5,
                 seed: Optional[int] = 0,
                 normalize: Optional[bool] = True):
        """RISE BlackBox Explainer

        Args:
            num_masks (int, optional): number of generated masks to aggregate
            num_cells (int, optional): number of cells for low-dimensional RISE
                random mask that later will be up-scaled to the model input size
            prob (float, optional): with prob p, a low-res cell is set to 1;
                otherwise, it's 0. Default: ``0.5``.
            seed (int, optional): Seed for random mask generation.
            normalize (bool, optional): Whether to normalize output or not.
        """
        super().__init__(model)
        self.input_size = model.inputs["data"].shape[-2:]
        self.num_masks = num_masks
        self.num_cells = num_cells
        self.prob = prob
        self.seed = seed
        self.normalize = normalize

    def explain(
        self,
        data: np.ndarray,
        target_explain_group: Optional[TargetExplainGroup] = None,
        explain_targets: Optional[List[int]] = None,
        post_processing_parameters: PostProcessParameters = PostProcessParameters(),
    ) -> ExplainResult:
        """Explain the input in black box mode.

        :param data: Data to explain.
        :type data: np.ndarray
        :param target_explain_group: Defines targets to explain: all classes, only predicted classes, etc.
        :type target_explain_group: TargetExplainGroup
        :param explain_targets: Provides list of custom targets, optional.
        :type explain_targets: Optional[List[int]]
        :param post_processing_parameters: Parameters that define post-processing.
        :type post_processing_parameters: PostProcessParameters
        """
        raw_saliency_map = self._generate_saliency_map(data)

        resized_data = self._resize_input(data)
        predicted_classes = self._model(resized_data)[0]
        cls_result = ClassificationResult(predicted_classes, raw_saliency_map, np.ndarray(0), np.ndarray(0))
        
        target_explain_group = self._get_target_explain_group(target_explain_group)
        explain_result = ExplainResult(cls_result, target_explain_group, explain_targets, self._labels)

        processed_explain_result = self._get_processed_explain_result(
            explain_result, data, post_processing_parameters
        )

        return processed_explain_result

    def _generate_saliency_map(self, data):
        """Generate RISE saliency map
        Returns:
            sal (np.ndarray): saliency map for each class

        """
        cell_size = np.ceil(np.array(self.input_size) / self.num_cells)
        up_size = np.array((self.num_cells + 1) * cell_size, dtype=np.uint32)
        rand_generator = np.random.default_rng(seed=self.seed)

        resized_data = self._resize_input(data)

        sal_maps = []
        for _ in tqdm(range(0, self.num_masks), desc="Explaining"):
            mask = self._generate_mask(cell_size, up_size, rand_generator)
            # Add channel dimensions for masks
            masked = np.expand_dims(mask, axis=2) * resized_data
            scores = self._model(masked).raw_scores
            sal = scores.reshape(-1, 1, 1) * mask
            sal_maps.append(sal)
        sal_maps = np.sum(sal_maps, axis=0)

        if self.normalize:
            sal_maps = self._normalize_saliency_maps(sal_maps)
        sal_maps = np.expand_dims(sal_maps, axis=0)
        return sal_maps

    def _generate_mask(self, cell_size, up_size, rand_generator):
        """Generate masks for RISE
            cell_size (int): calculated size of one cell for low-dimensional RISE
            up_size (int): increased cell size to crop
            rand_generator (np.random.generator): generator with fixed seed to generate random masks  
        Returns:
            masks (np.array): self.num_masks float masks from 0 to 1 with size of input model

        """
        grid_size = (self.num_cells, self.num_cells)
        grid = rand_generator.random(grid_size) < self.prob
        grid = grid.astype(np.float32)

        # Random shifts
        x = rand_generator.integers(0, cell_size[0])
        y = rand_generator.integers(0, cell_size[1])
        # Linear up-sampling and cropping
        upsampled_mask = cv2.resize(grid, up_size, interpolation=cv2.INTER_LINEAR)
        mask = upsampled_mask[x : x + self.input_size[0], y : y + self.input_size[1]]

        return mask

    def _resize_input(self, image):
        image = cv2.resize(image, self.input_size, cv2.INTER_LINEAR)
        return image

    def _normalize_saliency_maps(self, saliency_map):
        n, h, w = saliency_map.shape
        saliency_map = saliency_map.reshape((n, h * w))
        min_values = np.min(saliency_map, axis=-1)
        max_values = np.max(saliency_map, axis=-1)
        saliency_map = (
                255 * (saliency_map - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        )
        saliency_map = saliency_map.reshape((n, h, w)).astype(np.uint8)
        return saliency_map


class DRISEExplainer(BlackBoxExplainer):
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError
