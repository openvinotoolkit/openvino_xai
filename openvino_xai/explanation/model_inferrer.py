# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np
import openvino.runtime as ov

from openvino_xai.explanation.utils import InferenceResult
from openvino_xai.common.utils import has_xai, SALIENCY_MAP_OUTPUT_NAME


class ActivationType(Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


def softmax(x):
    """Compute softmax values of x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    """Compute sigmoid values of x."""
    return 1 / (1 + np.exp(-x))


class ClassificationModelInferrer:
    """
    Example of the custom model inferrer implementation. Provided just for a reference.
    Should be replaced with the user defined model inferrer.

    :param model: IR model.
    :type model: ov.Model
    """

    def __init__(
        self,
        model: ov.Model,
        input_size: Tuple[int, int] = (224, 224),
        change_channel_order: bool = False,
        mean: Optional[Union[np.ndarray, List[float]]] = None,
        std: Optional[Union[np.ndarray, List[float]]] = None,
        activation: ActivationType = ActivationType.SOFTMAX,
        output_name: Optional[str] = None,
    ):
        self.compiled_model = ov.Core().compile_model(model, "CPU")
        self.has_xai = has_xai(model)
        self.input_size = input_size
        self.change_channel_order = change_channel_order
        self.mean = mean if mean is not None else np.array([0.0, 0.0, 0.0])
        self.std = std if std is not None else np.array([1.0, 1.0, 1.0])
        self.activation = activation

        if output_name is not None:
            self.output_name = output_name
        else:
            for output in model.outputs:
                names = output.get_names()
                name = next(iter(names))
                if len(output.get_partial_shape()) == 2 and name != SALIENCY_MAP_OUTPUT_NAME:
                    self.output_name = name
                    break

    def __call__(self, x: np.ndarray) -> InferenceResult:
        x = self.preprocess(x)
        x = self.infer(x)
        x = self.postprocess(x)
        return x

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        # Change color channel order
        if self.change_channel_order:
            x = x[:, :, ::-1]
        # Resize to imagenet image shape.
        x = cv2.resize(src=x, dsize=self.input_size)
        # Normalize
        x = (x - self.mean) / self.std
        # Reshape to model input shape.
        x = x.transpose((2, 0, 1))
        x = np.expand_dims(x, 0)
        return x

    def infer(self, x: np.ndarray) -> ov.utils.data_helpers.wrappers.OVDict:
        # Inference call
        result_infer = self.compiled_model([x])
        return result_infer

    def postprocess(self, x: ov.utils.data_helpers.wrappers.OVDict) -> InferenceResult:
        prediction = x[self.output_name]
        # Process model prediction
        if self.activation != ActivationType.NONE:
            if self.activation == ActivationType.SOFTMAX:
                prediction = softmax(prediction)
            elif self.activation == ActivationType.SIGMOID:
                prediction = sigmoid(prediction)

        # Create inference result object
        inference_result = self._create_inference_result(x, prediction)
        return inference_result

    def _create_inference_result(self, raw_output, prediction_processed):
        if self.has_xai:
            saliency_map = raw_output[SALIENCY_MAP_OUTPUT_NAME]
            return InferenceResult(prediction_processed, saliency_map)
        else:
            return InferenceResult(prediction_processed)
