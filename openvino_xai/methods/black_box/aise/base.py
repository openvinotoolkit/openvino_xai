# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import openvino.runtime as ov
from scipy.optimize import direct

from openvino_xai.common.utils import IdentityPreprocessFN, is_bhwc_layout
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod


class AISEBase(BlackBoxXAIMethod, ABC):
    """
    AISE explains models in black-box mode using
    AISE: Adaptive Input Sampling for Explanation of Black-box Models
    (TODO (negvet): add link to the paper.)

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

        self.data_preprocessed = None
        self.target: int | None = None
        self.num_iterations_per_kernel: int | None = None
        self.kernel_widths: List[float] | np.ndarray | None = None
        self._current_kernel_width: float | None = None
        self.solver_epsilon = 0.1
        self.locally_biased = False
        self.kernel_params_hist: Dict = collections.defaultdict(list)
        self.pred_score_hist: Dict = collections.defaultdict(list)
        self.input_size: Tuple[int, int] | None = None
        self._mask_generator: GaussianPerturbationMask | None = None
        self.bounds = None
        self.preservation = True
        self.deletion = True

        if prepare_model:
            self.prepare_model()

    def _run_synchronous_explanation(self) -> np.ndarray:
        for kernel_width in self.kernel_widths:
            self._current_kernel_width = kernel_width
            self._run_optimization()
        return self._kernel_density_estimation()

    def _run_optimization(self):
        """Run DIRECT optimizer by default."""
        _ = direct(
            func=self._objective_function,
            bounds=self.bounds,
            eps=self.solver_epsilon,
            maxfun=self.num_iterations_per_kernel,
            locally_biased=self.locally_biased,
        )

    def _objective_function(self, args) -> float:
        """
        Objective function to optimize (to find a global minimum).
        Hybrid (dual) paradigm supporte two sub-objectives:
            - preservation
            - deletion
        """
        mh, mw = args
        kernel_params = (mh, mw, self._current_kernel_width)
        self.kernel_params_hist[self._current_kernel_width].append(kernel_params)

        kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
        kernel_mask = np.clip(kernel_mask, 0, 1)
        if is_bhwc_layout(self.data_preprocessed):
            kernel_mask = np.expand_dims(kernel_mask, 2)

        pred_loss_preserve = 0.0
        if self.preservation:
            data_perturbed_preserve = self.data_preprocessed * kernel_mask
            pred_loss_preserve = self._get_loss(data_perturbed_preserve)

        pred_loss_delete = 0.0
        if self.deletion:
            data_perturbed_delete = self.data_preprocessed * (1 - kernel_mask)
            pred_loss_delete = self._get_loss(data_perturbed_delete)

        loss = pred_loss_preserve - pred_loss_delete

        self.pred_score_hist[self._current_kernel_width].append(pred_loss_preserve - pred_loss_delete)

        loss *= -1  # Objective: minimize
        return loss

    @abstractmethod
    def _get_loss(self, data_perturbed: np.array) -> float:
        pass

    def _kernel_density_estimation(self) -> np.ndarray:
        """Aggregate the result per kernel with KDE."""
        saliency_map_per_kernel = np.zeros((len(self.kernel_widths), self.input_size[0], self.input_size[1]))
        for kernel_index, kernel_width in enumerate(self.kernel_widths):
            kernel_masks_weighted = np.zeros(self.input_size)
            for i in range(self.num_iterations_per_kernel):
                kernel_params = self.kernel_params_hist[kernel_width][i]
                kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
                score = self.pred_score_hist[kernel_width][i]
                kernel_masks_weighted += kernel_mask * score
            kernel_masks_weighted_max = kernel_masks_weighted.max()
            if kernel_masks_weighted_max > 0:
                kernel_masks_weighted = kernel_masks_weighted / kernel_masks_weighted_max
            saliency_map_per_kernel[kernel_index] = kernel_masks_weighted

        saliency_map = saliency_map_per_kernel.sum(axis=0)
        saliency_map /= saliency_map.max()
        return saliency_map


class GaussianPerturbationMask:
    """
    Perturbation mask generator.
    """

    def __init__(self, input_size: Tuple[int, int]):
        h = np.linspace(0, 1, input_size[0])
        w = np.linspace(0, 1, input_size[1])
        self.h, self.w = np.meshgrid(w, h)

    def _get_2d_gaussian(self, gauss_params: Tuple[float, float, float]) -> np.ndarray:
        mh, mw, sigma = gauss_params
        A = 1 / (2 * math.pi * sigma * sigma)
        B = (self.h - mh) ** 2 / (2 * sigma**2)
        C = (self.w - mw) ** 2 / (2 * sigma**2)
        return A * np.exp(-(B + C))

    def generate_kernel_mask(self, gauss_param: Tuple[float, float, float], scale: float = 1.0):
        """
        Generates 2D gaussian mask.
        """
        gaussian = self._get_2d_gaussian(gauss_param)
        return (gaussian / gaussian.max()) * scale
