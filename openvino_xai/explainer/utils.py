# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
from openvino.runtime.utils.data_helpers.wrappers import OVDict


def get_explain_target_indices(
    target_explain_labels: List[int | str],
    label_names: List[str] | None = None,
) -> List[int]:
    """
    Returns indices to be explained.

    :param target_explain_labels: List of custom labels to explain, optional. Can be list of integer indices (int),
        or list of names (str) from label_names.
    :type target_explain_labels: List[int | str]
    :param label_names: List of all label names.
    :type label_names: List[str] | None
    """
    if isinstance(target_explain_labels[0], int):
        return target_explain_labels  # type: ignore

    if not isinstance(target_explain_labels[0], str):
        raise ValueError(f"Explain labels expected to be int or str, but got {type(target_explain_labels[0])}")

    if not label_names:
        raise ValueError("Label names should be provided when target_explain_labels contain string names.")

    # Assuming len(target_explain_labels) << len(label_names)
    target_explain_indices = []
    for label_index, label in enumerate(label_names):
        if label in target_explain_labels:
            target_explain_indices.append(label_index)

    if len(target_explain_labels) != len(target_explain_indices):
        raise ValueError("No all label names found in label_names. Check spelling.")

    return target_explain_indices


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


def postprocess_fn(x: OVDict, logit_name="logits") -> np.ndarray:
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
