# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Mapping, Tuple

import cv2
import numpy as np
import openvino.runtime as ov
from tqdm import tqdm

from openvino_xai.common.utils import IdentityPreprocessFN, is_bhwc_layout, scaling
from openvino_xai.methods.base import Prediction
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod, Preset
from openvino_xai.methods.black_box.utils import check_classification_output


class RISE(BlackBoxXAIMethod):
    """RISE explains classification models in black-box mode using
    'RISE: Randomized Input Sampling for Explanation of Black-box Models' paper
    (https://arxiv.org/abs/1806.07421).

    postprocess_fn expected to return one container with scores. With batch dimention equals to one.

    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Post-processing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[Mapping], np.ndarray]
    :param preprocess_fn: Pre-processing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param device_name: Device type name.
    :type device_name: str
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        postprocess_fn: Callable[[Mapping], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        prepare_model: bool = True,
    ):
        super().__init__(
            model=model, postprocess_fn=postprocess_fn, preprocess_fn=preprocess_fn, device_name=device_name
        )
        self.num_masks: int | None = None
        self.num_cells: int | None = None

        if prepare_model:
            self.prepare_model()

    def generate_saliency_map(
        self,
        data: np.ndarray,
        target_indices: List[int] | None = None,
        preset: Preset = Preset.BALANCE,
        num_masks: int | None = None,
        num_cells: int | None = None,
        prob: float = 0.5,
        seed: int = 0,
        scale_output: bool = True,
    ) -> np.ndarray | Dict[int, np.ndarray]:
        """
        Generates inference result of the RISE algorithm.

        :param data: Input image.
        :type data: np.ndarray
        :param target_indices: List of target indices to explain.
        :type target_indices: List[int]
        :param preset: Speed-Quality preset, defines predefined configurations that manage speed-quality tradeoff.
        :type preset: Preset
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

        self.num_masks, self.num_cells = self._preset_parameters(preset, num_masks, num_cells)

        saliency_maps = self._run_synchronous_explanation(
            data_preprocessed,
            target_indices,
            prob,
            seed,
        )

        if isinstance(saliency_maps, np.ndarray):
            if scale_output:
                saliency_maps = scaling(saliency_maps)
            saliency_maps = np.expand_dims(saliency_maps, axis=0)
        elif isinstance(saliency_maps, dict):
            for target in saliency_maps:
                if scale_output:
                    saliency_maps[target] = scaling(saliency_maps[target])
        return saliency_maps

    @staticmethod
    def _preset_parameters(
        preset: Preset,
        num_masks: int | None = None,
        num_cells: int | None = None,
    ) -> Tuple[int, int]:
        if preset == Preset.SPEED:
            num_masks_ = 1000
            num_cells_ = 4
        elif preset == Preset.BALANCE:
            num_masks_ = 5000
            num_cells_ = 8
        elif preset == Preset.QUALITY:
            num_masks_ = 10000
            num_cells_ = 12
        else:
            raise ValueError(f"Preset {preset} is not supported.")

        if num_masks is None:
            num_masks = num_masks_
        if num_cells is None:
            num_cells = num_cells_

        return num_masks, num_cells

    def _run_synchronous_explanation(
        self,
        data_preprocessed: np.ndarray,
        target_classes: List[int] | None,
        prob: float,
        seed: int,
    ) -> np.ndarray:
        input_size = data_preprocessed.shape[1:3] if is_bhwc_layout(data_preprocessed) else data_preprocessed.shape[2:4]

        logits = self.get_logits(data_preprocessed)

        if target_classes is None:
            num_classes = logits.shape[1]
            num_targets = num_classes
        else:
            num_targets = len(target_classes)

        rand_generator = np.random.default_rng(seed=seed)

        saliency_maps = np.zeros((num_targets, input_size[0], input_size[1]))
        for _ in tqdm(range(0, self.num_masks), desc="Explaining in synchronous mode"):
            mask = self._generate_mask(input_size, self.num_cells, prob, rand_generator)
            # Add channel dimensions for masks
            if is_bhwc_layout(data_preprocessed):
                masked = np.expand_dims(mask, 2) * data_preprocessed
            else:
                masked = mask * data_preprocessed

            forward_output = self.model_forward(masked, preprocess=False)
            raw_scores = self.postprocess_fn(forward_output)
            check_classification_output(raw_scores)

            sal = self._get_scored_mask(raw_scores, mask, target_classes)
            saliency_maps += sal

        if target_classes is not None:
            saliency_maps = self._reformat_as_dict(saliency_maps, target_classes)
            for target in target_classes:
                self.predictions[target] = Prediction(
                    label=target,
                    score=logits[0][target],
                )
        return saliency_maps

    @staticmethod
    def _get_scored_mask(raw_scores: np.ndarray, mask: np.ndarray, target_classes: List[int] | None) -> np.ndarray:
        if target_classes is not None:
            return np.take(raw_scores, target_classes).reshape(-1, 1, 1) * mask
        else:
            return raw_scores.reshape(-1, 1, 1) * mask

    @staticmethod
    def _reformat_as_dict(
        saliency_maps: np.ndarray,
        target_classes: List[int] | None,
    ) -> np.ndarray:
        return {target_class: saliency_map for target_class, saliency_map in zip(target_classes, saliency_maps)}

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
