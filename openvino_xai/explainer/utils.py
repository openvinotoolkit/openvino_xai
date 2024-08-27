# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Mapping, Tuple

import cv2
import numpy as np

from openvino_xai.common.utils import sigmoid, softmax


def convert_targets_to_numpy(targets):
    targets = np.asarray(targets)
    if targets.ndim > 1:
        raise ValueError(f"targets expected to be at most 1-dimentional, but got {targets.ndim}.")
    return np.atleast_1d(targets)


def get_target_indices(
    targets: np.ndarray | List[int | str],
    label_names: List[str] | None = None,
) -> List[int]:
    """
    Returns indices to be explained.

    :param targets: List of custom labels to explain, optional. Can be list of integer indices (int),
        or list of names (str) from label_names.
    :type targets: np.ndarray | List[int | str]
    :param label_names: List of all label names.
    :type label_names: List[str] | None
    """
    targets = convert_targets_to_numpy(targets)

    if not isinstance(targets[0], str) and np.issubdtype(targets[0], np.integer):
        return targets  # type: ignore

    if not isinstance(targets[0], str):
        raise ValueError(f"Explain labels expected to be int or str, but got {type(targets[0])}")

    if not label_names:
        raise ValueError("Label names should be provided when targets contain string names.")

    # Assuming len(targets) << len(label_names)
    target_indices = []
    for label_index, label in enumerate(label_names):
        if label in targets:
            target_indices.append(label_index)

    if len(targets) != len(target_indices):
        raise ValueError("No all label names found in label_names. Check spelling.")

    return target_indices


def explains_all(targets: List[int | str] | int | str):
    """
    Defines reserved conditions for explaining all classes/labels.
    Introduced under the assumption that it is not gonna be frequently used.
    """
    if isinstance(targets, int) and targets == -1:
        return True
    if isinstance(targets, (np.ndarray, list)) and len(targets) == 1 and targets[0] == -1:
        return True
    if isinstance(targets, str) and targets == "-1":
        return True
    return False


def preprocess_fn(
    x: np.ndarray,
    change_channel_order: bool = False,
    input_size: Tuple[int, int] | None = None,
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
    change_channel_order: bool = False,
    input_size=None,
    mean: np.ndarray = np.array([0.0, 0.0, 0.0]),
    std: np.ndarray = np.array([1.0, 1.0, 1.0]),
    hwc_to_chw: bool = False,
    expand_zero_dim: bool = True,
) -> Callable[[Any], np.ndarray]:
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


class ActivationType(Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


def postprocess_fn(x: Mapping, logit_name="logits", activation: ActivationType = ActivationType.NONE) -> np.ndarray:
    """Postprocess function."""
    x = x.get(logit_name, x[0])  # Models from OVC has no output names at times
    if activation == ActivationType.SOFTMAX:
        return softmax(x)
    if activation == ActivationType.SIGMOID:
        return sigmoid(x)
    return x


def get_postprocess_fn(
    logit_name="logits", activation: ActivationType = ActivationType.NONE
) -> Callable[[], np.ndarray]:
    """Returns partially initialized postprocess_fn."""
    return partial(postprocess_fn, logit_name=logit_name, activation=activation)


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
