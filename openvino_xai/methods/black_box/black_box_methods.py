# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Tuple

import cv2
import numpy as np
import openvino.runtime as ov
from tqdm import tqdm

from openvino_xai.common.utils import IdentityPreprocessFN, scale
from openvino_xai.methods.method_base import MethodBase


class BlackBoxXAIMethodBase(MethodBase):
    """Base class for methods that explain model in Black-Box mode."""


class RISE(BlackBoxXAIMethodBase):
    """RISE explains classification models in black-box mode using RISE (https://arxiv.org/abs/1806.07421).

    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Preprocessing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray]
    :param preprocess_fn: Preprocessing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        postprocess_fn: Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        prepare_model: bool = True,
    ):
        super().__init__(model=model, preprocess_fn=preprocess_fn)
        self.postprocess_fn = postprocess_fn

        if prepare_model:
            self.prepare_model()

    def prepare_model(self) -> ov.Model:
        self.load_model()
        return self._model

    def generate_saliency_map(
        self,
        data: np.ndarray,
        explain_target_indices: List[int] | None = None,
        num_masks: int = 5000,
        num_cells: int = 8,
        prob: float = 0.5,
        seed: int = 0,
        scale_output: bool = True,
    ):
        """
        Generates inference result of the RISE algorithm.

        :param data: Input image.
        :type data: np.ndarray
        :param explain_target_indices: List of target indices to explain.
        :type explain_target_indices: List[int]
        :param num_masks: Number of generated masks to aggregate.
        :type num_masks: int
        :param num_cells: Number of cells for low-dimensional RISE
            random mask that later will be up-scaled to the model input size.
        :type num_cells: int
        :param prob: With prob p, a low-res cell is set to 1;
            otherwise, it's 0. Default: ``0.5``.
        :type prob: float
        :param seed: Seed for random mask generation.
        :type seed: int
        :param scale_output: Whether to scale output or not.
        :type scale_output: bool
        """
        data_preprocessed = self.preprocess_fn(data)

        saliency_maps = self._run_synchronous_explanation(
            data_preprocessed,
            explain_target_indices,
            num_masks,
            num_cells,
            prob,
            seed,
        )

        if scale_output:
            saliency_maps = scale(saliency_maps)
        saliency_maps = np.expand_dims(saliency_maps, axis=0)
        return saliency_maps

    def _run_synchronous_explanation(
        self,
        data_preprocessed: np.ndarray,
        target_classes: List[int] | None,
        num_masks: int,
        num_cells: int,
        prob: float,
        seed: int,
    ) -> np.ndarray:
        _, _, height, width = data_preprocessed.shape
        input_size = height, width

        forward_output = self.model_forward(data_preprocessed, preprocess=False)
        logits = self.postprocess_fn(forward_output)
        _, num_classes = logits.shape

        if target_classes is None:
            num_targets = num_classes
        else:
            num_targets = len(target_classes)

        rand_generator = np.random.default_rng(seed=seed)

        sal_maps = np.zeros((num_targets, input_size[0], input_size[1]))
        for _ in tqdm(range(0, num_masks), desc="Explaining in synchronous mode"):
            mask = self._generate_mask(input_size, num_cells, prob, rand_generator)
            # Add channel dimensions for masks
            masked = mask * data_preprocessed

            forward_output = self.model_forward(masked, preprocess=False)
            raw_scores = self.postprocess_fn(forward_output)

            sal = self._get_scored_mask(raw_scores, mask, target_classes)
            sal_maps += sal

        if target_classes is not None:
            sal_maps = self._reconstruct_sparce_saliency_map(sal_maps, num_classes, input_size, target_classes)
        return sal_maps

    @staticmethod
    def _get_scored_mask(raw_scores: np.ndarray, mask: np.ndarray, target_classes: List[int] | None) -> np.ndarray:
        if target_classes:
            return np.take(raw_scores, target_classes).reshape(-1, 1, 1) * mask
        else:
            return raw_scores.reshape(-1, 1, 1) * mask

    @staticmethod
    def _reconstruct_sparce_saliency_map(
        sal_maps: np.ndarray, num_classes: int, input_size, target_classes: List[int] | None
    ) -> np.ndarray:
        # TODO: see if np.put() or other alternatives works faster (requires flatten array)
        sal_maps_tmp = sal_maps
        sal_maps = np.zeros((num_classes, input_size[0], input_size[1]))
        for i, sal in enumerate(sal_maps_tmp):
            sal_maps[target_classes[i]] = sal
        return sal_maps

    @staticmethod
    def _generate_mask(input_size: Tuple[int, int], num_cells: int, prob: float, rand_generator) -> np.ndarray:
        """Generate masks for RISE
        Returns:
            mask (np.array): float mask from 0 to 1 with size of model input
        """
        cell_size = np.ceil(np.array(input_size) / num_cells)
        up_size = np.array((num_cells + 1) * cell_size, dtype=np.uint32)

        grid_size = (num_cells, num_cells)
        grid = rand_generator.random(grid_size) < prob
        grid = grid.astype(np.float32)

        # Random shifts
        x = rand_generator.integers(0, cell_size[0])
        y = rand_generator.integers(0, cell_size[1])
        # Linear up-sampling and cropping
        upsampled_mask = cv2.resize(grid, up_size, interpolation=cv2.INTER_CUBIC)
        mask = upsampled_mask[x : x + input_size[0], y : y + input_size[1]]
        mask = np.clip(mask, 0, 1)
        return mask
