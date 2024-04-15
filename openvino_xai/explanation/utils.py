# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from functools import partial
from typing import Optional, Union, List, Tuple, Callable

import cv2
import numpy as np
import openvino.runtime as ov

from openvino_xai.explanation.explanation_parameters import TargetExplainGroup


def select_target_indices(
    target_explain_group: TargetExplainGroup,
    explain_target_indices: Optional[Union[List[int], np.ndarray]] = None,
    total_num_targets: Optional[int] = None,
) -> Union[List[int], np.ndarray]:
    """
    Selects target indices.

    :param target_explain_group: Target explain group.
    :type target_explain_group: TargetExplainGroup
    :param explain_target_indices: Target explain indices.
    :type explain_target_indices: Optional[Union[list, np.ndarray]]
    :param total_num_targets: Total number of targets.
    :type total_num_targets: Optional[int]
    """

    if target_explain_group == TargetExplainGroup.CUSTOM:
        if explain_target_indices is None:
            raise ValueError(f"Explain targets has to be provided for {target_explain_group}.")
        if not total_num_targets:
            raise ValueError("total_num_targets has to be provided.")
        if not all(0 <= target_index <= (total_num_targets - 1) for target_index in explain_target_indices):
            raise ValueError(f"All targets explanation indices have to be in range 0..{total_num_targets - 1}.")
        return explain_target_indices

    raise ValueError(f"Unsupported target_explain_group: {target_explain_group}")


def preprocess_fn(
        x: np.ndarray,
        change_channel_order: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        mean: np.ndarray = np.array([0.0, 0.0, 0.0]),
        std: np.ndarray = np.array([1.0, 1.0, 1.0]),
        hwc_to_chw: bool = False,
        expand_zero_dim: bool = True,
) -> np.ndarray:
    """Preprocess function."""
    # Change color channel order
    if change_channel_order:
        x = x[:, :, ::-1]

    # Resize
    if input_size:
        x = cv2.resize(src=x, dsize=input_size)

    # Normalize
    x = (x - mean) / std

    # Change layout HxWxC => CxHxW
    if hwc_to_chw:
        x = x.transpose((2, 0, 1))

    # Add batch dim
    if expand_zero_dim:
        x = np.expand_dims(x, 0)

    return x


def get_preprocess_fn(
        change_channel_order=False,
        input_size=None,
        mean=np.array([0.0, 0.0, 0.0]),
        std=np.array([1.0, 1.0, 1.0]),
        hwc_to_chw=False,
        expand_zero_dim=True,
) -> Callable[[ov.utils.data_helpers.wrappers.OVDict], np.ndarray]:
    """Returns partially initialized preprocess_fn."""
    return partial(
        preprocess_fn,
        change_channel_order=change_channel_order,
        input_size=input_size,
        mean=mean,
        std=std,
        hwc_to_chw=hwc_to_chw,
        expand_zero_dim=expand_zero_dim,
    )


def postprocess_fn(x: ov.utils.data_helpers.wrappers.OVDict, logit_name="logits") -> np.ndarray:
    """Postprocess function."""
    return x[logit_name]


def get_postprocess_fn(logit_name="logits") -> Callable[[], np.ndarray]:
    """Returns partially initialized postprocess_fn."""
    return partial(postprocess_fn, logit_name=logit_name)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values of x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid values of x."""
    return 1 / (1 + np.exp(-x))


class ActivationType(Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


def get_score(x: np.ndarray, index: int, activation: ActivationType = ActivationType.NONE):
    """Returns activated score at index."""
    if activation == ActivationType.SOFTMAX:
        x = softmax(x)
    if activation == ActivationType.SIGMOID:
        x = sigmoid(x)
    if len(x.shape) == 2:
        assert x.shape[0] == 1
        return x[0, index]
    return x[index]
