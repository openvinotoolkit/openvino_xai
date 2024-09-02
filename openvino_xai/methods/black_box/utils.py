# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np


def check_classification_output(x: np.ndarray) -> None:
    """Checks output of the postprocess function provided by the user (for classification talk)."""
    if not isinstance(x, np.ndarray):
        raise RuntimeError("Postprocess function should return numpy array.")
    if x.ndim != 2 or x.shape[0] != 1:
        raise RuntimeError("Postprocess function should return two dimentional numpy array with batch size of 1.")


def check_detection_output(x: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Checks output of the postprocess function provided by the user (for detection task)."""
    if not hasattr(x, "__len__"):
        raise RuntimeError("Postprocess function should return sized object.")

    if len(x) != 3:
        raise RuntimeError(
            "Postprocess function should return three containers: boxes (format: [x1, y1, x2, y2]), scores, labels."
        )

    for item in x:
        if not isinstance(item, np.ndarray):
            raise RuntimeError("Postprocess function should return numpy arrays.")
        if item.shape[0] != 1:
            raise RuntimeError("Postprocess function should return numpy arrays with batch size of 1.")

    boxes, scores, labels = x
    if boxes.ndim != 3:
        raise RuntimeError("Boxes should be three-dimentional [Batch, NumBoxes, BoxCoords].")
    if scores.ndim != 2:
        raise RuntimeError("Scores should be two-dimentional [Batch, Scores].")
    if labels.ndim != 2:
        raise RuntimeError("Labels should be two-dimentional  [Batch, Labels].")
