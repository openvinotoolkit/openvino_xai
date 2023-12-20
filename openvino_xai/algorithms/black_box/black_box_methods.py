# mypy: disable-error-code="union-attr"

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Union, Callable, Optional, Tuple, List

import cv2
import numpy as np
from openvino.model_api import models as mapi_models
from openvino.model_api.models import ClassificationModel
from openvino.model_api.pipelines import AsyncPipeline
from tqdm import tqdm

from openvino_xai.common.utils import has_xai, logger
from openvino_xai.explanation.explanation_parameters import TargetExplainGroup, SELECTED_TARGETS, ExplanationParameters
from openvino_xai.explanation.utils import InferenceResult, get_prediction_from_model_output, select_target_indices


class BlackBoxXAIMethodBase(ABC):
    """Base class for methods that explain model in Black-Box mode."""


class RISE(BlackBoxXAIMethodBase):
    """RISEExplainer explains classification models in black-box mode using RISE (https://arxiv.org/abs/1806.07421).

    :param model_inferrer: Callable model inferrer object.
    :type model_inferrer: Union[Callable[[np.ndarray], InferenceResult], mapi_models.Model]
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
    :param input_size: Model input size.
    :type input_size: Tuple[int]
    :param asynchronous_inference: Whether to run inference in asynchronous mode or not.
    :type asynchronous_inference: bool
    :param throughput_inference: Whether to run asynchronous inference in throughput mode or not.
    :type throughput_inference: bool
    :param normalize: Whether to normalize output or not.
    :type normalize: bool
    """

    def __init__(
        self,
        model_inferrer: Union[Callable[[np.ndarray], InferenceResult], mapi_models.Model],
        num_masks: int = 5000,
        num_cells: int = 8,
        prob: float = 0.5,
        seed: int = 0,
        input_size: Optional[Tuple[int]] = None,
        asynchronous_inference: bool = True,
        throughput_inference: bool = True,
        normalize: bool = True,
    ):
        self._model_mapi_wrapper = isinstance(model_inferrer, ClassificationModel)
        self._xai_branch_warning = (
            "Input model has XAI branch inserted, which might lead to additional "
            "computational overhead, due to computation in XAI head. "
            "Consider providing pure model w/o xai inserted "
            "to black-box explainer for better performance."
        )
        if self._model_mapi_wrapper:
            if has_xai(model_inferrer.inference_adapter.model):
                logger.warning(self._xai_branch_warning)
            if asynchronous_inference and throughput_inference:
                model_inferrer.inference_adapter.plugin_config.update({"PERFORMANCE_HINT": "THROUGHPUT"})
                model_inferrer.load(force=True)

        self._model_inferrer = model_inferrer
        self.num_masks = num_masks
        self.num_cells = num_cells
        self.prob = prob
        self.rand_generator = np.random.default_rng(seed=seed)

        if self._model_mapi_wrapper:
            assert len(model_inferrer.inputs) == 1, "Support only for models with single input."
            input_name = next(iter(model_inferrer.inputs))
            self.input_size = model_inferrer.inputs[input_name].shape[-2:]
        else:
            if input_size:
                self.input_size = input_size
            else:
                logger.warning(
                    "Input size is not provided, setting to (224, 224), which might be incorrect. "
                    "Provide input_size for reliable results."
                )
                self.input_size = (224, 224)

        # TODO: optimize for custom pipelines
        self.asynchronous_inference = asynchronous_inference if self._model_mapi_wrapper else False
        self.normalize = normalize

    def get_result(self, data: np.ndarray, explanation_parameters: ExplanationParameters = ExplanationParameters()):
        """Generates inference result of RISE algorithm."""
        model_output = self._model_inferrer(data)
        if isinstance(model_output, InferenceResult):
            if model_output.saliency_map is not None:
                logger.warning(self._xai_branch_warning)
        elif isinstance(model_output, mapi_models.ClassificationResult):
            if model_output.saliency_map.size != 0:
                logger.warning(self._xai_branch_warning)
        else:
            raise ValueError(
                f"Model output has to be ether "
                f"openvino_xai.explanation.utils.InferenceResult or "
                f"openvino.model_api.models.ClassificationResult, but got{type(model_output)}."
            )

        prediction_raw, target_classes = self._get_prediction_and_target_classes(model_output, explanation_parameters)
        num_classes = len(prediction_raw)
        saliency_map = self._generate_saliency_map(data, num_classes, target_classes)

        inference_result = InferenceResult(prediction_raw[None, ...], saliency_map)
        return inference_result

    def _get_prediction_and_target_classes(
        self,
        model_output: Union[InferenceResult, mapi_models.ClassificationResult],
        explanation_parameters: ExplanationParameters,
    ) -> Tuple:
        prediction, prediction_raw = get_prediction_from_model_output(
            model_output, explanation_parameters.confidence_threshold
        )
        if explanation_parameters.target_explain_group == TargetExplainGroup.PREDICTIONS:
            if not prediction:
                raise ValueError(
                    "TargetExplainGroup.PREDICTIONS requires predictions "
                    "to be available, but currently model has no predictions. "
                    "Try to: (1) adjust preprocessing, (2) use different input, "
                    "(3) decrease confidence threshold, (4) retrain/re-export the model, etc."
                )
        if prediction_raw.size == 0:
            raise ValueError(
                "RISEExplainer expects Model API wrapper to be created (via ClassificationModel.create_model()) "
                "with output_raw_scores configuration flag set to True."
            )
        num_classes = len(prediction_raw)

        if explanation_parameters.target_explain_group in SELECTED_TARGETS:
            target_classes = self._get_target_classes(
                num_classes,
                explanation_parameters.target_explain_group,
                explanation_parameters.custom_target_indices,
                prediction,
            )
        else:
            target_classes = None
        return prediction_raw, target_classes

    @staticmethod
    def _get_target_classes(
        num_classes: int,
        target_explain_group: TargetExplainGroup,
        explain_targets: Union[List[int], np.ndarray],
        prediction: Union[List[Tuple[int]], np.ndarray],
    ):
        prediction_indices = [pred[0] for pred in prediction]
        explain_target_indexes = select_target_indices(
            target_explain_group,
            prediction_indices,
            explain_targets,
            num_classes,
        )
        return explain_target_indexes

    def _generate_saliency_map(
        self, data: np.ndarray, num_classes: int, target_classes: Optional[List[int]]
    ) -> np.ndarray:
        """Generates RISE saliency map.

        Returns:
            saliency_maps (np.ndarray): saliency map for each class
        """

        resized_data = self._resize_input(data)
        if self.asynchronous_inference:
            saliency_maps = self._run_asynchronous_explanation(resized_data, num_classes, target_classes)
        else:
            saliency_maps = self._run_synchronous_explanation(resized_data, num_classes, target_classes)

        if self.normalize:
            saliency_maps = self._normalize_saliency_maps(saliency_maps)
        saliency_maps = np.expand_dims(saliency_maps, axis=0)
        return saliency_maps

    def _run_asynchronous_explanation(
        self, resized_data: np.ndarray, num_classes: int, target_classes: Optional[List[int]]
    ) -> np.ndarray:
        logger.info(
            f"RISE explains the model in asynchronous mode " f"with {self.num_masks} masks (inference calls)..."
        )
        masks = []
        async_pipeline = AsyncPipeline(self._model_inferrer)
        for i in range(self.num_masks):
            mask = self._generate_mask()
            masks.append(mask)
            # Add channel dimensions for masks
            masked = np.expand_dims(mask, axis=2) * resized_data
            async_pipeline.submit_data(masked, i)
        async_pipeline.await_all()

        if target_classes is None:
            num_targets = num_classes
        else:
            num_targets = len(target_classes)

        sal_maps = np.zeros((num_targets, self.input_size[0], self.input_size[1]))
        for j in range(self.num_masks):
            result, _ = async_pipeline.get_result(j)
            raw_scores = result.raw_scores
            sal = self._get_scored_mask(raw_scores, masks[j], target_classes)
            sal_maps += sal

        if target_classes is not None:
            sal_maps = self._reconstruct_sparce_saliency_map(sal_maps, num_classes, target_classes)
        return sal_maps

    def _run_synchronous_explanation(
        self, resized_data: np.ndarray, num_classes: int, target_classes: Optional[List[int]]
    ) -> np.ndarray:
        if target_classes is None:
            num_targets = num_classes
        else:
            num_targets = len(target_classes)

        sal_maps = np.zeros((num_targets, self.input_size[0], self.input_size[1]))
        for _ in tqdm(range(0, self.num_masks), desc="Explaining in synchronous mode"):
            mask = self._generate_mask()
            # Add channel dimensions for masks
            masked = np.expand_dims(mask, axis=2) * resized_data
            if self._model_mapi_wrapper:
                raw_scores = self._model_inferrer(masked).raw_scores
            else:
                raw_scores = self._model_inferrer(masked).prediction
            sal = self._get_scored_mask(raw_scores, mask, target_classes)
            sal_maps += sal

        if target_classes is not None:
            sal_maps = self._reconstruct_sparce_saliency_map(sal_maps, num_classes, target_classes)
        return sal_maps

    @staticmethod
    def _get_scored_mask(raw_scores: np.ndarray, mask: np.ndarray, target_classes: Optional[List[int]]) -> np.ndarray:
        if target_classes:
            return np.take(raw_scores, target_classes).reshape(-1, 1, 1) * mask
        else:
            return raw_scores.reshape(-1, 1, 1) * mask

    def _reconstruct_sparce_saliency_map(
        self, sal_maps: np.ndarray, num_classes: int, target_classes: Optional[List[int]]
    ) -> np.ndarray:
        # TODO: see if np.put() or other alternatives works faster (requires flatten array)
        sal_maps_tmp = sal_maps
        sal_maps = np.zeros((num_classes, self.input_size[0], self.input_size[1]))
        for i, sal in enumerate(sal_maps_tmp):
            sal_maps[target_classes[i]] = sal
        return sal_maps

    def _generate_mask(self) -> np.ndarray:
        """Generate masks for RISE
        Returns:
            mask (np.array): float mask from 0 to 1 with size of model input

        """
        cell_size = np.ceil(np.array(self.input_size) / self.num_cells)
        up_size = np.array((self.num_cells + 1) * cell_size, dtype=np.uint32)

        grid_size = (self.num_cells, self.num_cells)
        grid = self.rand_generator.random(grid_size) < self.prob
        grid = grid.astype(np.float32)

        # Random shifts
        x = self.rand_generator.integers(0, cell_size[0])
        y = self.rand_generator.integers(0, cell_size[1])
        # Linear up-sampling and cropping
        upsampled_mask = cv2.resize(grid, up_size, interpolation=cv2.INTER_CUBIC)
        mask = upsampled_mask[x : x + self.input_size[0], y : y + self.input_size[1]]
        mask = np.clip(mask, 0, 1)
        return mask

    def _resize_input(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, self.input_size, cv2.INTER_LINEAR)
        return image

    @staticmethod
    def _normalize_saliency_maps(saliency_map: np.ndarray) -> np.ndarray:
        n, h, w = saliency_map.shape
        saliency_map = saliency_map.reshape((n, h * w))
        min_values = np.min(saliency_map, axis=-1)
        max_values = np.max(saliency_map, axis=-1)
        saliency_map = 255 * (saliency_map - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        saliency_map = saliency_map.reshape((n, h, w)).astype(np.uint8)
        return saliency_map
