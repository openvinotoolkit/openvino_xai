# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict
from scipy.optimize import Bounds, direct

from openvino_xai.common.utils import (
    IdentityPreprocessFN,
    infer_size_from_image,
    logger,
    scaling,
    sigmoid,
)
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod, Preset


class AISE(BlackBoxXAIMethod):
    """AISE explains classification models in black-box mode using
    AISE: Adaptive Input Sampling for Explanation of Black-box Models
    (TODO (negvet): add link to the paper.)

    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Post-processing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[OVDict], np.ndarray]
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
        postprocess_fn: Callable[[OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        prepare_model: bool = True,
    ):
        super().__init__(model=model, preprocess_fn=preprocess_fn, device_name=device_name)
        self.postprocess_fn = postprocess_fn

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

        if prepare_model:
            self.prepare_model()

    def generate_saliency_map(  # type: ignore
        self,
        data: np.ndarray,
        target_indices: List[int] | None,
        preset: Preset = Preset.BALANCE,
        num_iterations_per_kernel: int | None = None,
        kernel_widths: List[float] | np.ndarray | None = None,
        solver_epsilon: float = 0.1,
        locally_biased: bool = False,
        scale_output: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Generates inference result of the AISE algorithm.
        Optimized for per class saliency map generation. Not effcient for large number of classes.

        :param data: Input image.
        :type data: np.ndarray
        :param target_indices: List of target indices to explain.
        :type target_indices: List[int]
        :param preset: Speed-Quality preset, defines predefined configurations that manage the speed-quality tradeoff.
        :type preset: Preset
        :param num_iterations_per_kernel: Number of iterations per kernel, defines compute budget.
        :type num_iterations_per_kernel: int
        :param kernel_widths: Kernel bandwidths.
        :type kernel_widths: List[float] | np.ndarray
        :param solver_epsilon: Solver epsilon of DIRECT optimizer.
        :type solver_epsilon: float
        :param locally_biased: Locally biased flag of DIRECT optimizer.
        :type locally_biased: bool
        :param scale_output: Whether to scale output or not.
        :type scale_output: bool
        """
        self.data_preprocessed = self.preprocess_fn(data)

        if target_indices is None:
            num_classes = self.get_num_classes(self.data_preprocessed)
            if num_classes > 10:
                logger.info(f"num_classes = {num_classes}, which might take significant time to process.")
            target_indices = list(range(num_classes))

        self.num_iterations_per_kernel, self.kernel_widths = self._preset_parameters(
            preset,
            num_iterations_per_kernel,
            kernel_widths,
        )

        self.solver_epsilon = solver_epsilon
        self.locally_biased = locally_biased

        self.input_size = infer_size_from_image(self.data_preprocessed)
        self._mask_generator = GaussianPerturbationMask(self.input_size)

        saliency_maps = {}
        for target in target_indices:
            self.kernel_params_hist = collections.defaultdict(list)
            self.pred_score_hist = collections.defaultdict(list)

            self.target = target
            saliency_map_per_target = self._run_synchronous_explanation()
            if scale_output:
                saliency_map_per_target = scaling(saliency_map_per_target)
            saliency_maps[target] = saliency_map_per_target
        return saliency_maps

    @staticmethod
    def _preset_parameters(
        preset: Preset,
        num_iterations_per_kernel: int | None,
        kernel_widths: List[float] | np.ndarray | None,
    ) -> Tuple[int, np.ndarray]:
        if preset == Preset.SPEED:
            iterations = 25
            widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.BALANCE:
            iterations = 50
            widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.QUALITY:
            iterations = 85
            widths = np.linspace(0.075, 0.25, 4)
        else:
            raise ValueError(f"Preset {preset} is not supported.")

        if num_iterations_per_kernel is None:
            num_iterations_per_kernel = iterations
        if kernel_widths is None:
            kernel_widths = widths
        return num_iterations_per_kernel, kernel_widths

    def _run_synchronous_explanation(self) -> np.ndarray:
        for kernel_width in self.kernel_widths:
            self._current_kernel_width = kernel_width
            self._run_optimization()
        return self._kernel_density_estimation()

    def _run_optimization(self):
        """Run DIRECT optimizer by default."""
        _ = direct(
            func=self._objective_function,
            bounds=Bounds([0.0, 0.0], [1.0, 1.0]),
            eps=self.solver_epsilon,
            maxfun=self.num_iterations_per_kernel,
            locally_biased=self.locally_biased,
        )

    def _objective_function(self, args) -> float:
        """
        Objective function to optimize (to find a global minimum).
        Hybrid (dual) paradigm adopted with two sub-objectives:
            - preservation
            - deletion
        """
        mh, mw = args
        kernel_params = (mh, mw, self._current_kernel_width)
        self.kernel_params_hist[self._current_kernel_width].append(kernel_params)

        kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
        kernel_mask = np.clip(kernel_mask, 0, 1)

        data_perturbed_preserve = self.data_preprocessed * kernel_mask
        pred_score_preserve = self._get_score(data_perturbed_preserve)

        data_perturbed_delete = self.data_preprocessed * (1 - kernel_mask)
        pred_score_delete = self._get_score(data_perturbed_delete)

        loss = pred_score_preserve - pred_score_delete

        self.pred_score_hist[self._current_kernel_width].append(pred_score_preserve - pred_score_delete)

        loss *= -1  # Objective: minimize
        return loss

    def _get_score(self, data_perturbed: np.array) -> float:
        """Get model prediction score for perturbed input."""
        x = self.model_forward(data_perturbed, preprocess=False)
        x = self.postprocess_fn(x)
        if np.max(x) > 1 or np.min(x) < 0:
            x = sigmoid(x)
        pred_scores = x.squeeze()  # type: ignore
        return pred_scores[self.target]

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
            kernel_masks_weighted = kernel_masks_weighted / kernel_masks_weighted.max()
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
