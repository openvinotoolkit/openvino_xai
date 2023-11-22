# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Tuple, List

import numpy as np

from openvino.model_api.models import ClassificationResult, DetectionResult

from openvino_xai.common.utils import logger
from openvino_xai.explanation.explanation_parameters import TargetExplainGroup


class InferenceResult:
    """Inference result (with batch dimensions).

    :param prediction: Model prediction.
    :type prediction: np.ndarray
    :param saliency_map: Generated saliency map.
    :type saliency_map: np.ndarray
    """

    def __init__(self, prediction: np.ndarray, saliency_map: Optional[np.ndarray] = None):
        self.prediction = prediction
        self.saliency_map = saliency_map


def get_prediction_from_model_output(
        inference_result: Union[InferenceResult, ClassificationResult, DetectionResult],
        confidence_threshold: float
) -> Tuple:
    """Generate prediction and prediction_raw."""
    # TODO: align between classification and detection
    if isinstance(inference_result, InferenceResult):
        check_prediction_from_model_output(inference_result.prediction)
        prediction_raw = inference_result.prediction
        prediction_raw = prediction_raw.squeeze()
        # TODO: add label to align with mapi
        prediction = [
            (index, score) for index, score in enumerate(prediction_raw) if score > confidence_threshold
        ]
    elif isinstance(inference_result, ClassificationResult):
        prediction, prediction_raw = inference_result.top_labels, inference_result.raw_scores
    elif isinstance(inference_result, DetectionResult):
        objects = inference_result.objects
        # Currently support only for DetClassProbabilityMapXAIMethod
        # TODO: generalize
        prediction = set((detection.id, None) for detection in objects)
        prediction_raw = None
    else:
        raise ValueError(f"inference result type {type(inference_result)} is not supported.")
    return prediction, prediction_raw


def check_prediction_from_model_output(prediction: np.ndarray) -> None:
    """Validates model prediction."""
    if not isinstance(prediction, np.ndarray):
        raise ValueError(f"Model prediction has to be np.ndarray, but got {type(prediction)}.")
    if prediction.size == 0:
        raise RuntimeError("Model output does not contain valid prediction.")
    if prediction.shape[0] > 1:
        raise RuntimeError("Batch size for raw prediction should be 1.")
    if not 0 <= prediction.min() <= prediction.max() <= 1:
        raise ValueError(f"Prediction has to be activated.")


def select_target_indices(
        target_explain_group: TargetExplainGroup,
        prediction_indices: Optional[Union[List[int], np.ndarray]] = None,
        explain_target_indices: Optional[Union[List[int], np.ndarray]] = None,
        total_num_targets: Optional[int] = None,
) -> Union[List[int], np.ndarray]:
    """
    Selects target indices.

    :param target_explain_group: Target explain group.
    :type target_explain_group: TargetExplainGroup
    :param prediction_indices: Prediction indices.
    :type prediction_indices: Optional[Union[list, np.ndarray]]
    :param explain_target_indices: Target explain indices.
    :type explain_target_indices: Optional[Union[list, np.ndarray]]
    :param total_num_targets: Total number of targets.
    :type total_num_targets: Optional[int]
    """

    if target_explain_group == TargetExplainGroup.PREDICTIONS:
        if not prediction_indices:
            raise ValueError(
                f"{target_explain_group} requires predictions "
                "to be available, but currently model has no predictions. "
                "Try to: (1) adjust preprocessing, (2) use different input, "
                "(3) increase confidence threshold, (4) retrain/re-export the model, etc."
            )
        if explain_target_indices:
            logger.warning(
                f"Explain targets do NOT have to be provided for "
                f"{target_explain_group}. Model prediction is used "
                f"to retrieve explanation targets."
            )
        return prediction_indices

    if target_explain_group == TargetExplainGroup.CUSTOM:
        if explain_target_indices is None:
            raise ValueError(f"Explain targets has to be provided for {target_explain_group}.")
        if not total_num_targets:
            raise ValueError(f"total_num_targets has to be provided.")
        if not all(0 <= target_index <= (total_num_targets - 1) for target_index in explain_target_indices):
            raise ValueError(
                f"All targets explanation indices have to be in range 0..{total_num_targets - 1}."
            )
        return explain_target_indices
