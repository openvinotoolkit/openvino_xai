# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Callable, Dict, List, Tuple

import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict
from scipy.optimize import Bounds

from openvino_xai.common.utils import (
    IdentityPreprocessFN,
    infer_size_from_image,
    logger,
    scaling,
    sigmoid,
)
from openvino_xai.methods.base import Prediction
from openvino_xai.methods.black_box.aise.base import AISEBase, GaussianPerturbationMask
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.methods.black_box.utils import check_classification_output


class AISEClassification(AISEBase):
    """
    AISE for classification models.

    postprocess_fn expected to return one container with scores. With batch dimention equals to one.

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
        super().__init__(
            model=model,
            postprocess_fn=postprocess_fn,
            preprocess_fn=preprocess_fn,
            device_name=device_name,
            prepare_model=prepare_model,
        )
        self.bounds = Bounds([0.0, 0.0], [1.0, 1.0])
        self.num_iterations_per_kernel: int | None = None
        self.kernel_widths: List[float] | np.ndarray | None = None

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

        logits = self.get_logits(self.data_preprocessed)
        if target_indices is None:
            num_classes = logits.shape[1]
            target_indices = list(range(num_classes))
            if len(target_indices) > 10:
                logger.info(f"{len(target_indices)} targets to process, which might take significant time.")

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
        self.predictions = {}
        for target in target_indices:
            self.kernel_params_hist = collections.defaultdict(list)
            self.pred_score_hist = collections.defaultdict(list)

            self.target = target
            saliency_map_per_target = self._run_synchronous_explanation()
            if scale_output:
                saliency_map_per_target = scaling(saliency_map_per_target)
            saliency_maps[target] = saliency_map_per_target
            self.predictions[target] = Prediction(
                label=target,
                score=logits[0][target],
            )
        return saliency_maps

    @staticmethod
    def _preset_parameters(
        preset: Preset,
        num_iterations_per_kernel: int | None,
        kernel_widths: List[float] | np.ndarray | None,
    ) -> Tuple[int, np.ndarray]:
        if preset == Preset.SPEED:
            iterations = 20
            widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.BALANCE:
            iterations = 50
            widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.QUALITY:
            iterations = 50
            widths = np.linspace(0.075, 0.25, 5)
        else:
            raise ValueError(f"Preset {preset} is not supported.")

        if num_iterations_per_kernel is None:
            num_iterations_per_kernel = iterations
        if kernel_widths is None:
            kernel_widths = widths
        return num_iterations_per_kernel, kernel_widths

    def _get_loss(self, data_perturbed: np.ndarray) -> float:
        """Get loss for perturbed input."""
        x = self.model_forward(data_perturbed, preprocess=False)
        x = self.postprocess_fn(x)
        check_classification_output(x)

        if np.max(x) > 1 or np.min(x) < 0:
            x = sigmoid(x)
        return x[0][self.target]
