# from abc import ABC
# from abc import abstractmethod
# from typing import Dict, Any, Optional, List

# import cv2
# import numpy as np
# import openvino
# from PIL import Image
# from tqdm import tqdm
# from skimage.transform import resize

# from openvino.model_api.models import ClassificationResult

# from openvino_xai.model import XAIModel, XAIClassificationModel
# from openvino_xai.saliency_map import ExplainResult, PostProcessor, TargetExplainGroup
# from openvino_xai.utils import logger


# class Explainer(ABC):
#     """A base interface for explainer."""

#     def __init__(self, model: openvino.model_api.models.Model):
#         self._model = model
#         self._explain_method = self._model.explain_method if hasattr(self._model, "explain_method") else None
#         self._labels = self._model.labels

#     @abstractmethod
#     def explain(self, data: np.ndarray) -> ExplainResult:
#         """Explain the input."""
#         raise NotImplementedError

#     def _get_target_explain_group(self, target_explain_group):
#         if target_explain_group:
#             if self._explain_method:
#                 assert (
#                     target_explain_group in self._explain_method.supported_target_explain_groups
#                 ), f"Provided target_explain_group {target_explain_group} is not supported by the explain method."
#             return target_explain_group
#         else:
#             if self._explain_method:
#                 return self._explain_method.default_target_explain_group
#             else:
#                 raise ValueError(
#                     "Model with XAI branch was created outside of Openvino-XAI library. "
#                     "Please explicitly provide target_explain_group to the explain call."
#                 )


# class WhiteBoxExplainer(Explainer):
#     """Explainer explains models with XAI branch injected."""

#     def explain(
#         self,
#         data: np.ndarray,
#         target_explain_group: Optional[TargetExplainGroup] = None,
#         explain_targets: Optional[List[int]] = None,
#         post_processing_parameters: Optional[Dict[str, Any]] = None,
#     ) -> ExplainResult:
#         """Explain the input in white box mode."""
#         raw_result = self._model(data)

#         target_explain_group = self._get_target_explain_group(target_explain_group)
#         explain_result = ExplainResult(raw_result, target_explain_group, explain_targets, self._labels)

#         post_processing_parameters = post_processing_parameters or {}
#         post_processor = PostProcessor(explain_result, data, **post_processing_parameters)
#         explain_result = post_processor.postprocess()
#         return explain_result


# class BlackBoxExplainer(Explainer):
#     """Base class for explainers that consider model as a black-box."""


# class RISEExplainer(BlackBoxExplainer):
#     def __init__(self, model, num_masks=2000, num_cells=8, prob=0.5):
#         """RISE BlackBox Explainer

#         Args:
#             num_masks (int, optional): number of generated masks to aggregate
#             num_cells (int, optional): number of cells for low-dimentaional RISE
#                 random mask that later will be upscaled to the model input size
#             prob (float, optional): with prob p, a low-res cell is set to 1;
#                 otherwise, it's 0. Default: ``0.5``.

#         """
#         super().__init__(model)
#         self.input_size = model.inputs["data"].shape[-2:]
#         self.num_masks = num_masks
#         self.num_cells = num_cells
#         self.prob = prob

#     def explain(
#         self,
#         data,
#         target_explain_group: Optional[TargetExplainGroup] = None,
#         explain_targets: Optional[List[int]] = None,
#         post_processing_parameters: Optional[Dict[str, Any]] = None,
#     ):
#         """Explain the input."""
#         cls_result = self._generate_saliency_map(data)
        
#         target_explain_group = self._get_target_explain_group(target_explain_group)
#         explain_result = ExplainResult(cls_result, target_explain_group, explain_targets, self._labels)

#         post_processing_parameters = post_processing_parameters or {}
#         post_processor = PostProcessor(explain_result, data, **post_processing_parameters)
#         explain_result = post_processor.postprocess()
#         return explain_result

#     def _generate_saliency_map(self, data):
#         """Generate RISE saliency map
#         Returns:
#             cls_result (ClassificationResult): classification result with labels and saliency map

#         """
#         self._generate_masks()
#         resized_data = self._resize_input(data)
#         cls_result = self._model(resized_data)
#         print(cls_result[0])

#         preds = []
#         for i in tqdm(range(0, self.num_masks), desc="Explaining"):
#             # Add channel dimentions for masks
#             masked = np.expand_dims(self.masks[i], axis=2) * resized_data
#             scores = [self._model(masked).raw_scores]
#             preds.append(scores)
#         preds = np.concatenate(preds)

#         sal = preds.T.dot(self.masks.reshape(self.num_masks, -1)).reshape(-1, *self.input_size)
#         sal = sal / self.num_masks / self.prob
#         sal = np.expand_dims(sal, axis=0)
#         cls_result = ClassificationResult(cls_result[0], sal, np.ndarray(0), np.ndarray(0))
#         return cls_result

#     def _generate_masks(self):
#         """Generate masks for RISE
#         Returns:
#             masks (np.array): self.num_masks float masks from 0 to 1 with size of input model

#         """
#         cell_size = np.ceil(np.array(self.input_size) / self.num_cells)
#         up_size = np.array((self.num_cells + 1) * cell_size, dtype=np.uint32)

#         rng = np.random.default_rng(seed=42)
#         grid = rng.random(self.num_masks, self.num_cells, self.num_cells) < self.prob
#         grid = grid.astype("float32")

#         self.masks = np.empty((self.num_masks, *self.input_size))

#         for i in tqdm(range(self.num_masks), desc="Generating filters"):
#             # Random shifts
#             x = np.random.randint(0, cell_size[0])
#             y = np.random.randint(0, cell_size[1])
#             # Linear upsampling and cropping
#             self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode="reflect", anti_aliasing=False)[
#                 x : x + self.input_size[0], y : y + self.input_size[1]
#             ]

#     def _resize_input(self, image):
#         image = cv2.resize(image, self.input_size, Image.BILINEAR)
#         return image


# class DRISEExplainer(BlackBoxExplainer):
#     def explain(self, data: np.ndarray) -> ExplainResult:
#         """Explain the input."""
#         raise NotImplementedError

