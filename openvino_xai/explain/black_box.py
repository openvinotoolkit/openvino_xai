# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from openvino.model_api.models import ClassificationModel, ClassificationResult
from openvino.model_api.pipelines import AsyncPipeline

from openvino_xai.explain.base import Explainer
from openvino_xai.model import XAIModel
from openvino_xai.parameters import PostProcessParameters
from openvino_xai.saliency_map import ExplainResult, TargetExplainGroup, SELECTED_CLASSES
from openvino_xai.utils import logger


class BlackBoxExplainer(Explainer):
    """Base class for explainers that consider model as a black-box."""


class RISEExplainer(BlackBoxExplainer):
    """RISEExplainer explains classification models in black-box mode using RISE (https://arxiv.org/abs/1806.07421).

    :param model: ModelAPI wrapper.
    :type model: openvino.model_api.models.ClassificationModel
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

    def __init__(self,
                 model: ClassificationModel,
                 num_masks: int = 5000,
                 num_cells: int = 8,
                 prob: float = 0.5,
                 seed: int = 0,
                 input_size: Optional[Tuple[int]] = None,
                 asynchronous_inference: bool = True,
                 throughput_inference: bool = True,
                 normalize: bool = True):
        if not isinstance(model, ClassificationModel):
            raise ValueError(f"Input model suppose to be openvino.model_api.models.ClassificationModel instance, "
                             f"but got {type(model)}.")
        if XAIModel.has_xai(model.inference_adapter.model):
            logger.warning(f"Input model has XAI branch inserted, which might lead to additional "
                           f"computational overhead, due to computation in XAI head. "
                           f"Consider providing pure openvino.model_api.models.ClassificationModel "
                           f"to black-box explainer for better performance.")

        if asynchronous_inference and throughput_inference:
            model.inference_adapter.plugin_config.update({"PERFORMANCE_HINT": "THROUGHPUT"})
            model.load(force=True)

        super().__init__(model)
        self.num_masks = num_masks
        self.num_cells = num_cells
        self.prob = prob
        self.rand_generator = np.random.default_rng(seed=seed)

        if input_size:
            self.input_size = input_size
        else:
            assert len(model.inputs) == 1, "Support only for models with single input."
            input_name = next(iter(model.inputs))
            self.input_size = model.inputs[input_name].shape[-2:]

        self.asynchronous_inference = asynchronous_inference
        self.normalize = normalize

        # TODO: Temporary workaround, fix in https://github.com/intel-sandbox/openvino_xai/issues/19
        explain_method_namespace = {
            "default_target_explain_group": TargetExplainGroup.PREDICTED_CLASSES,
            "supported_target_explain_groups": {TargetExplainGroup.ALL_CLASSES,
                                                TargetExplainGroup.PREDICTED_CLASSES,
                                                TargetExplainGroup.CUSTOM_CLASSES}
        }
        self._explain_method = type("DummyExplainMethodClass", (object,), explain_method_namespace)()

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
        resized_data = self._resize_input(data)
        if self._model.hierarchical:
            hierarchical_info = self._model.hierarchical_info["cls_heads_info"]
        else:
            hierarchical_info = None
    
        result = self._model(resized_data)
        if result.raw_scores.size == 0:
            raise ValueError(
                "RISEExplainer expects Model API wrapper to be created (via ClassificationModel.create_model()) "
                "with output_raw_scores configuration flag set to True."
            )
        predictions = result.top_labels
        num_classes = len(result.raw_scores)
        target_classes = self._get_target_classes(num_classes, target_explain_group, explain_targets, predictions)

        raw_saliency_map = self._generate_saliency_map(data, num_classes, target_classes)
        cls_result = ClassificationResult(predictions, raw_saliency_map, np.ndarray(0), np.ndarray(0))
        
        target_explain_group = self._get_target_explain_group(target_explain_group)
        raw_explain_result = ExplainResult(cls_result, target_explain_group, explain_targets, self._labels, hierarchical_info)

        processed_explain_result = self._get_processed_explain_result(
            raw_explain_result, data, post_processing_parameters
        )
        return processed_explain_result

    @staticmethod
    def _get_target_classes(num_classes, target_explain_group, explain_targets, predictions):
        if target_explain_group in SELECTED_CLASSES:
            if target_explain_group == TargetExplainGroup.PREDICTED_CLASSES:
                assert explain_targets is None, f"For {TargetExplainGroup.PREDICTED_CLASSES} explain group, " \
                                                f"targets will be estimated from the model prediction. " \
                                                f"explain_targets should not be provided."
                target_classes = [prediction[0] for prediction in predictions]
            elif target_explain_group == TargetExplainGroup.CUSTOM_CLASSES:
                assert (
                    explain_targets is not None
                ), f"Explain targets has to be provided for {target_explain_group}."
                assert (
                    all(0 <= target <= num_classes - 1 for target in explain_targets)
                ), f"For class-wise targets, all explain targets has to be in range 0..{num_classes - 1}"
                target_classes = explain_targets
            else:
                raise ValueError(f"Target explain group {target_explain_group} is not supported.")
        else:
            target_classes = None
        return target_classes

    def _generate_saliency_map(
            self, data: np.ndarray, num_classes: int, target_classes: Optional[List[int]]
    ) -> np.ndarray:
        """Generate RISE saliency map
        Returns:
            sal (np.ndarray): saliency map for each class

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
    ):
        logger.info(f"RISEExplainer explains the model in asynchronous mode "
                    f"with {self.num_masks} masks (inference calls)...")
        masks = []
        async_pipeline = AsyncPipeline(self._model)
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
    ):
        if target_classes is None:
            num_targets = num_classes
        else:
            num_targets = len(target_classes)

        sal_maps = np.zeros((num_targets, self.input_size[0], self.input_size[1]))
        for _ in tqdm(range(0, self.num_masks), desc="Explaining in synchronous mode"):
            mask = self._generate_mask()
            # Add channel dimensions for masks
            masked = np.expand_dims(mask, axis=2) * resized_data
            raw_scores = self._model(masked).raw_scores
            sal = self._get_scored_mask(raw_scores, mask, target_classes)
            sal_maps += sal

        if target_classes is not None:
            sal_maps = self._reconstruct_sparce_saliency_map(sal_maps, num_classes, target_classes)
        return sal_maps

    @staticmethod
    def _get_scored_mask(raw_scores: np.ndarray, mask: np.ndarray, target_classes: Optional[List[int]]):
        if target_classes:
            return np.take(raw_scores, target_classes).reshape(-1, 1, 1) * mask
        else:
            return raw_scores.reshape(-1, 1, 1) * mask

    def _reconstruct_sparce_saliency_map(self, sal_maps, num_classes, target_classes):
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
        saliency_map = (
                255 * (saliency_map - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        )
        saliency_map = saliency_map.reshape((n, h, w)).astype(np.uint8)
        return saliency_map


class DRISEExplainer(BlackBoxExplainer):
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError
