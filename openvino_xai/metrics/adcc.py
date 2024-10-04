from typing import Any, Dict, List

import numpy as np
from scipy.stats import pearsonr

from openvino_xai.common.utils import scaling
from openvino_xai.explainer.explanation import ONE_MAP_LAYOUTS, Explanation
from openvino_xai.metrics.base import BaseMetric


class ADCC(BaseMetric):
    """
    Implementation of the Average Drop-Coherence-Complexity (ADCC) metric by Poppi, Samuele, et al 2021.

    References:
        1) Poppi, Samuele, et al. "Revisiting the evaluation of class activation mapping for explainability:
            A novel metric and experimental analysis." Proceedings of the IEEE/CVF Conference on
            Computer Vision and Pattern Recognition. 2021.
        2) Reference implementation:
            https://github.com/aimagelab/ADCC/
    """

    def __init__(self, model, preprocess_fn, postprocess_fn, explainer, device_name="CPU", **kwargs: Any):
        super().__init__(
            model=model, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn, device_name=device_name
        )
        self.explainer = explainer
        self.black_box_kwargs = kwargs

    def average_drop(self, masked_image: np.ndarray, class_idx: int, model_output: np.ndarray) -> float:
        """
        Measures the average percentage drop in confidence for the target class when the model sees only the
        explanation map (image masked with saliency map), instead of the full image.
        The less the better.
        """
        confidence_on_input = model_output[class_idx]
        prediction_on_saliency_map = self.model_predict(masked_image)
        confidence_on_saliency_map = prediction_on_saliency_map[class_idx]

        return max(0.0, confidence_on_input - confidence_on_saliency_map) / confidence_on_input

    def coherency(self, saliency_map: np.ndarray, masked_image: np.ndarray, class_idx: int, image: np.ndarray) -> float:
        """
        Measures the coherency of the saliency map. The explanation map (image masked with saliency map) should
        contain all the relevant features that explain a prediction and should remove useless features in a coherent way.
        Saliency map and saliency map of exlanation map should be similar.
        The more the better.
        """
        saliency_map_masked_image = self.explainer(
            masked_image, targets=class_idx, colormap=False, scaling=False, **self.black_box_kwargs
        )
        saliency_map_masked_image = list(saliency_map_masked_image.saliency_map.values())[0]  # only one target
        saliency_map_masked_image = scaling(saliency_map_masked_image, cast_to_uint8=False, max_value=1)

        A, B = saliency_map.flatten(), saliency_map_masked_image.flatten()
        # Pearson correlation coefficient
        y, _ = pearsonr(A, B)
        y = (y + 1) / 2

        return y

    @staticmethod
    def complexity(saliency_map: np.ndarray) -> float:
        """
        Measures the complexity of the saliency map. Less important pixels -> less complexity.
        Defined as L1 norm of the saliency map.
        The less the better.
        """
        return saliency_map.sum() / (saliency_map.shape[-1] * saliency_map.shape[-2])

    def __call__(self, saliency_map: np.ndarray, class_idx: int, input_image: np.ndarray) -> Dict[str, float]:
        """
        Calculate the ADCC metric for a given saliency map and class index.
        The more the better.

        Parameters:
        :param saliency_map: Saliency map for class_idx class (H, W).
        :type saliency_map: np.ndarray
        :param class_idx: The class index of saliency map.
        :type class_idx: int
        :param input_image: The input image to the model (H, W, C).
        :type input_image: np.ndarray

        Returns:
        :return: A dictionary containing the ADCC, coherency, complexity, and average drop metrics.
        :rtype: Dict[str, float]
        """
        # Scale the saliency map to [0, 1] range.
        if not (0 <= np.min(saliency_map) and np.max(saliency_map) <= 1):
            saliency_map = scaling(saliency_map, cast_to_uint8=False, max_value=1)

        model_output = self.model_predict(input_image)
        masked_image = input_image * saliency_map[:, :, None]
        class_idx = np.argmax(model_output) if class_idx is None else class_idx

        avgdrop = self.average_drop(masked_image, class_idx, model_output)
        coh = self.coherency(saliency_map, masked_image, class_idx, input_image)
        com = self.complexity(saliency_map)

        adcc = 3 / (1 / coh + 1 / (1 - com) + 1 / (1 - avgdrop))
        return {"adcc": adcc, "coherency": coh, "complexity": com, "average_drop": avgdrop}

    def evaluate(
        self, explanations: List[Explanation], input_images: List[np.ndarray], **kwargs: Any
    ) -> Dict[str, float]:
        """
        Evaluate the ADCC metric over a list of explanations and input images.

        Parameters:
        :param explanations: A list of explanations for each image.
        :type explanations: List[Explanation]
        :param input_images: A list of input images.
        :type input_images: List[np.ndarray]

        Returns:
        :return: A dictionary containing the average ADCC, coherency, complexity, and average drop metrics.
        :rtype: Dict[str, float]
        """
        results = []
        for input_image, explanation in zip(input_images, explanations):
            for class_idx, saliency_map in explanation.saliency_map.items():
                target_idx = None if explanation.layout in ONE_MAP_LAYOUTS else int(class_idx)
                metric_dict = self(saliency_map, target_idx, input_image)
                results.append(
                    [
                        metric_dict["coherency"],
                        metric_dict["complexity"],
                        metric_dict["average_drop"],
                    ]
                )
        coherency, complexity, average_drop = np.mean(np.array(results), axis=0)
        adcc = 3 / (1 / coherency + 1 / (1 - complexity) + 1 / (1 - average_drop))
        return {"adcc": adcc, "coherency": coherency, "complexity": complexity, "average_drop": average_drop}
