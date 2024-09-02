from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import numpy as np
import openvino as ov

from openvino_xai.common.utils import IdentityPreprocessFN
from openvino_xai.explainer.explanation import Explanation


class BaseMetric(ABC):
    """Base class for XAI quality metric."""

    def __init__(
        self,
        model: ov.Model = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        postprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
        device_name: str = "CPU",
    ):
        # Pass model_predict to class initialization directly?
        self.model = model
        self.model_compiled = ov.Core().compile_model(model=model, device_name=device_name)
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def model_predict(self, input: np.ndarray) -> np.ndarray:
        logits = self.model_compiled([self.preprocess_fn(input)])
        logits = self.postprocess_fn(logits)[0]
        return logits

    @abstractmethod
    def __call__(self, saliency_map, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Calculate the metric for the single saliency map"""

    @abstractmethod
    def evaluate(self, explanations: List[Explanation], *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Evaluate the quality of saliency maps over the list of images"""
