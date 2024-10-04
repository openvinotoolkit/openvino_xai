from typing import Any, Dict, List, Tuple

import numpy as np

from openvino_xai.explainer.explanation import ONE_MAP_LAYOUTS, Explanation
from openvino_xai.metrics.base import BaseMetric


def AUC(arr: np.array) -> float:
    """
    Returns normalized Area Under Curve of the array.
    """
    return np.abs((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1))


class InsertionDeletionAUC(BaseMetric):
    """
    Implementation of the Insertion and Deletion AUC by Petsiuk et al. 2018.

    References:
        Petsiuk, Vitali, Abir Das, and Kate Saenko. "Rise: Randomized input sampling
        for explanation of black-box models." arXiv preprint arXiv:1806.07421 (2018).
    """

    @staticmethod
    def step_image_insertion_deletion(
        num_pixels: int, sorted_indices: Tuple[np.ndarray, np.ndarray], input_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return insertion/deletion image based on number of pixels to add/delete on this step.
        """
        # Values to start
        image_insertion = np.full_like(input_image, 0)
        image_deletion = input_image.copy()

        x_indices = sorted_indices[0][:num_pixels]
        y_indices = sorted_indices[1][:num_pixels]

        # Insert the image on the places of the important pixels
        image_insertion[x_indices, y_indices] = input_image[x_indices, y_indices]
        # Remove image pixels on the places of the important pixels
        image_deletion[x_indices, y_indices] = 0
        return image_insertion, image_deletion

    def __call__(
        self, saliency_map: np.ndarray, class_idx: int, input_image: np.ndarray, steps: int = 30, **kwargs: Any
    ) -> Dict[str, float]:
        """
        Calculate the Insertion and Deletion AUC metrics for one saliency map for one class.

        Parameters:
        :param saliency_map: Importance scores for each pixel (H, W).
        :type saliency_map: np.ndarray
        :param class_idx: The class of saliency map to evaluate.
        :type class_idx: int
        :param input_image: The input image to the model (H, W, C).
        :type input_image: np.ndarray
        :param steps: Number of steps for inserting pixels.
        :type steps: int

        Returns:
        :return: A dictionary containing the AUC scores for insertion and deletion scores.
        :rtype: Dict[str, float]
        """

        class_idx = np.argmax(self.model_predict(input_image)) if class_idx is None else class_idx

        # Sort pixels by descending importance to find the most important pixels
        sorted_indices = np.argsort(-saliency_map.flatten())
        sorted_indices = np.unravel_index(sorted_indices, saliency_map.shape)

        insertion_scores, deletion_scores = [], []
        for i in range(steps + 1):
            num_pixels = int(i * len(sorted_indices[0]) / steps)
            step_image_insertion, step_image_deletion = self.step_image_insertion_deletion(
                num_pixels, sorted_indices, input_image
            )
            # Predict on masked image
            insertion_scores.append(self.model_predict(step_image_insertion)[class_idx])
            deletion_scores.append(self.model_predict(step_image_deletion)[class_idx])
        insertion = AUC(np.array(insertion_scores))
        deletion = AUC(np.array(deletion_scores))
        return {"insertion": insertion, "deletion": deletion}

    def evaluate(
        self, explanations: List[Explanation], input_images: List[np.ndarray], steps: int, **kwargs: Any
    ) -> Dict[str, float]:
        """
        Evaluate the insertion and deletion AUC over the list of images and its saliency maps.

        :param explanations: List of explanation objects containing saliency maps.
        :type explanations: List[Explanation]
        :param input_images: List of input images as numpy arrays.
        :type input_images: List[np.ndarray]
        :param steps: Number of steps for the insertion and deletion process.
        :type steps: int

        :return: A Dict containing the mean insertion AUC, mean deletion AUC, and their difference (delta) as values.
        :rtype: float
        """
        results = []
        for input_image, explanation in zip(input_images, explanations):
            for class_idx, saliency_map in explanation.saliency_map.items():
                target_idx = None if explanation.layout in ONE_MAP_LAYOUTS else int(class_idx)
                metric_dict = self(saliency_map, target_idx, input_image, steps)
                results.append([metric_dict["insertion"], metric_dict["deletion"]])

        insertion, deletion = np.mean(np.array(results), axis=0)
        delta = insertion - deletion
        return {"insertion": insertion, "deletion": deletion, "delta": delta}
