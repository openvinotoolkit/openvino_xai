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
)
from openvino_xai.methods.base import Prediction
from openvino_xai.methods.black_box.aise.base import AISEBase, GaussianPerturbationMask
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.methods.black_box.utils import check_detection_output


class AISEDetection(AISEBase):
    """
    AISE for detection models.

    postprocess_fn expected to return three containers: boxes (format: [x1, y1, x2, y2]), scores, labels. With batch dimention equals to one.

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
        self.deletion = False
        self.predictions = {}
        self.num_iterations_per_kernel: int | None = None
        self.divisors: List[float] | np.ndarray | None = None

    def generate_saliency_map(  # type: ignore
        self,
        data: np.ndarray,
        target_indices: List[int] | None,
        preset: Preset = Preset.BALANCE,
        num_iterations_per_kernel: int | None = None,
        divisors: List[float] | np.ndarray | None = None,
        solver_epsilon: float = 0.05,
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
        :param divisors: List of dividors, used to derive kernel widths in an adaptive manner.
        :type divisors: List[float] | np.ndarray
        :param solver_epsilon: Solver epsilon of DIRECT optimizer.
        :type solver_epsilon: float
        :param locally_biased: Locally biased flag of DIRECT optimizer.
        :type locally_biased: bool
        :param scale_output: Whether to scale output or not.
        :type scale_output: bool
        """
        # TODO (negvet): support custom bboxes (not predicted ones)

        self.data_preprocessed = self.preprocess_fn(data)
        forward_output = self.model_forward(self.data_preprocessed, preprocess=False)

        # postprocess_fn expected to return three containers: boxes (x1, y1, x2, y2), scores, labels.
        output = self.postprocess_fn(forward_output)
        check_detection_output(output)
        boxes, scores, labels = output
        boxes, scores, labels = boxes[0], scores[0], labels[0]

        if target_indices is None:
            num_boxes = len(boxes)
            if num_boxes > 10:
                logger.info(f"num_boxes = {num_boxes}, which might take significant time to process.")
            target_indices = list(range(num_boxes))

        self.num_iterations_per_kernel, self.divisors = self._preset_parameters(
            preset,
            num_iterations_per_kernel,
            divisors,
        )

        self.solver_epsilon = solver_epsilon
        self.locally_biased = locally_biased

        self.input_size = infer_size_from_image(self.data_preprocessed)
        original_size = infer_size_from_image(data)
        self._mask_generator = GaussianPerturbationMask(self.input_size)

        saliency_maps = {}
        self.predictions = {}
        for target in target_indices:
            self.target_box = boxes[target]
            self.target_label = labels[target]

            if self.target_box[0] >= self.target_box[2] or self.target_box[1] >= self.target_box[3]:
                continue

            self.kernel_params_hist = collections.defaultdict(list)
            self.pred_score_hist = collections.defaultdict(list)

            self._process_box()
            saliency_map_per_target = self._run_synchronous_explanation()
            if scale_output:
                saliency_map_per_target = scaling(saliency_map_per_target)
            saliency_maps[target] = saliency_map_per_target

            self._update_predictions(boxes, scores, labels, target, original_size)
        return saliency_maps

    @staticmethod
    def _preset_parameters(
        preset: Preset,
        num_iterations_per_kernel: int | None,
        divisors: List[float] | np.ndarray | None,
    ) -> Tuple[int, np.ndarray]:
        if preset == Preset.SPEED:
            iterations = 20
            divs = np.linspace(7, 1, 3)
        elif preset == Preset.BALANCE:
            iterations = 50
            divs = np.linspace(7, 1, 3)
        elif preset == Preset.QUALITY:
            iterations = 50
            divs = np.linspace(8, 1, 5)
        else:
            raise ValueError(f"Preset {preset} is not supported.")

        if num_iterations_per_kernel is None:
            num_iterations_per_kernel = iterations
        if divisors is None:
            divisors = divs
        return num_iterations_per_kernel, divisors

    def _process_box(self, padding_coef: float = 0.5) -> None:
        target_box_scaled = [
            self.target_box[0] / self.input_size[1],  # x1
            self.target_box[1] / self.input_size[0],  # y1
            self.target_box[2] / self.input_size[1],  # x2
            self.target_box[3] / self.input_size[0],  # y2
        ]
        box_width = target_box_scaled[2] - target_box_scaled[0]
        box_height = target_box_scaled[3] - target_box_scaled[1]
        self._min_box_size = min(box_width, box_height)
        self.kernel_widths = [self._min_box_size / div for div in self.divisors]

        x_from = max(target_box_scaled[0] - box_width * padding_coef, 0.0)
        x_to = min(target_box_scaled[2] + box_width * padding_coef, 1.0)
        y_from = max(target_box_scaled[1] - box_height * padding_coef, 0.0)
        y_to = min(target_box_scaled[3] + box_height * padding_coef, 1.0)
        self.bounds = Bounds([x_from, y_from], [x_to, y_to])

    def _get_loss(self, data_perturbed: np.array) -> float:
        """Get loss for perturbed input."""
        forward_output = self.model_forward(data_perturbed, preprocess=False)
        boxes, scores, labels = self.postprocess_fn(forward_output)
        boxes, scores, labels = boxes[0], scores[0], labels[0]

        loss = 0
        for box, score, label in zip(boxes, scores, labels):
            if label == self.target_label:
                loss = max(loss, self._iou(self.target_box, box) * score)
        return loss

    @staticmethod
    def _iou(box1: np.ndarray | List[float], box2: np.ndarray | List[float]) -> float:
        box1 = np.asarray(box1)
        box2 = np.asarray(box2)
        tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
        br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
        intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
        area1 = np.prod(box1[2:] - box1[:2])
        area2 = np.prod(box2[2:] - box2[:2])
        return intersection / (area1 + area2 - intersection)

    def _update_predictions(
        self,
        boxes: np.ndarray | List,
        scores: np.ndarray | List[float],
        labels: np.ndarray | List[int],
        target: int,
        original_size: Tuple[int, int],
    ) -> None:
        x1, y1, x2, y2 = boxes[target]
        width_scale = original_size[1] / self.input_size[1]
        height_scale = original_size[0] / self.input_size[0]
        x1, x2 = x1 * width_scale, x2 * width_scale
        y1, y2 = y1 * height_scale, y2 * height_scale
        self.predictions[target] = Prediction(
            label=labels[target],
            score=scores[target],
            bounding_box=(x1, y1, x2, y2),
        )
