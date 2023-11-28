# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import math

import cv2
import numpy as np
import openvino.runtime as ov


from openvino_xai.explanation.utils import InferenceResult
from openvino_xai.common.utils import has_xai, SALIENCY_MAP_OUTPUT_NAME


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class ClassificationModelInferrer:
    """
    Example of the custom model inferrer implementation. Provided just for a reference.
    Should be replaced with the user defined model inferrer.

    :param model: IR model.
    :type model: ov.Model
    """

    def __init__(self, model: ov.Model, sigmoid: bool = True):
        self.compiled_model = ov.Core().compile_model(model, "CPU")
        self.has_xai = has_xai(model)
        self.sigmoid = sigmoid  # disable activation if it is already inserted into the model graph

    @staticmethod
    def preprocess(x: np.ndarray) -> np.ndarray:
        # Resize to imagenet image shape.
        x = cv2.resize(src=x, dsize=(224, 224))
        # Reshape to model input shape.
        x = x.transpose((2, 0, 1))
        x = np.expand_dims(x, 0)
        return x

    def infer(self, x: np.ndarray) -> ov.utils.data_helpers.wrappers.OVDict:
        result_infer = self.compiled_model([x])
        return result_infer

    def postprocess(self, x: ov.utils.data_helpers.wrappers.OVDict) -> InferenceResult:
        # Process model prediction
        prediction = x["logits"]
        if self.sigmoid:
            sigmoidv = np.vectorize(sigmoid)
            prediction = sigmoidv(prediction)

        # Create inference result object
        inference_result = self._create_inference_result(x, prediction)
        return inference_result

    def _create_inference_result(self, raw_output, prediction_processed):
        if self.has_xai:
            saliency_map = raw_output[SALIENCY_MAP_OUTPUT_NAME]
            return InferenceResult(prediction_processed, saliency_map)
        else:
            return InferenceResult(prediction_processed)

    def __call__(self, x: np.ndarray) -> InferenceResult:
        x = self.preprocess(x)
        x = self.infer(x)
        x = self.postprocess(x)
        return x
