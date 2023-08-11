from abc import ABC
from abc import abstractmethod
from typing import Dict, Any, Optional, List

import numpy as np
import openvino

from openvino_xai.model import XAIModel, XAIClassificationModel
from openvino_xai.saliency_map import ExplainResult, PostProcessor, TargetExplainGroup
from openvino_xai.utils import logger


class Explainer(ABC):
    """A base interface for explainer."""

    def __init__(self, model: openvino.model_api.models.Model):
        self._model = model
        self._explain_method = self._model.explain_method if hasattr(self._model, "explain_method") else None
        self._labels = self._model.labels

    @abstractmethod
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError

    def _get_target_explain_group(self, target_explain_group):
        if target_explain_group:
            if self._explain_method:
                assert target_explain_group in self._explain_method.supported_target_explain_groups, \
                    f"Provided target_explain_group {target_explain_group} is not supported by the explain method."
            return target_explain_group
        else:
            if self._explain_method:
                return self._explain_method.default_target_explain_group
            else:
                raise ValueError("Model with XAI branch was created outside of Openvino-XAI library. "
                                 "Please explicitly provide target_explain_group to the explain call.")


class WhiteBoxExplainer(Explainer):
    """Explainer explains models with XAI branch injected."""

    def explain(
            self,
            data: np.ndarray,
            target_explain_group: Optional[TargetExplainGroup] = None,
            explain_targets: Optional[List[int]] = None,
            post_processing_parameters: Optional[Dict[str, Any]] = None,
    ) -> ExplainResult:
        """Explain the input in white box mode."""
        raw_result = self._model(data)

        target_explain_group = self._get_target_explain_group(target_explain_group)
        explain_result = ExplainResult(raw_result, target_explain_group, explain_targets, self._labels)

        post_processing_parameters = post_processing_parameters or {}
        post_processor = PostProcessor(explain_result, data, **post_processing_parameters)
        explain_result = post_processor.postprocess()
        return explain_result


class BlackBoxExplainer(Explainer):
    """Base class for explainers that consider model as a black-box."""


class RISEExplainer(BlackBoxExplainer):
    def __init__(self, model):
        super().__init__(model)
        # self._model = model
        self.input_size = model.inputs['data'].shape[-2:]
        self.num_masks = 10
        self.num_cells = 8
        self.prob = 0.5

    def explain(self, data):

        """Explain the input."""
        
        data_size = data.shape
        self.input_size = data_size[:2]
        self._generate_masks(self.num_masks, self.num_cells, self.prob)

        preds = []

        # data = np.transpose(data, (2, 0, 1)) # c, h, w

        # masked = data * self.masks
        for i in tqdm(range(0, self.num_masks), desc='Explaining'):
            masked = np.expand_dims(self.masks[i], axis=2) * data
            pred = self._model(masked)
            if 'raw_scores' in pred:
                scores = pred['raw_scores']
            else:
                # softmax is needed
                scores = pred['logits']
            preds.append(scores)
        preds = np.concatenate(preds)
        sal = preds.T.dot(self.masks.reshape(self.num_masks, -1)).reshape(-1, *self.input_size)
        sal = sal / self.num_masks / self.prob
        sal = np.expand_dims(sal, axis=0)
        # sal = self._processor.postprocess(sal)
        sal = self._postprocess(sal)

        return sal
    
    @staticmethod
    def _postprocess(saliency_map):
        min_soft_score = np.min(saliency_map)
        max_soft_score = np.max(saliency_map)
        saliency_map = 255.0 / (max_soft_score + 1e-12) * (saliency_map - min_soft_score)
        return saliency_map


    def _generate_masks(self, N, s, p1, savepath='masks.npy'):
        
        from skimage.transform import resize
        # import torch
        """ Generate masks for RISE
            Args:
                N: number of masks
                s: number of cells for one spatial dimension
                    in low-res RISE random mask
                p (float, optional): with prob p, a low-res cell is set to 0;
                    otherwise, it's 1. Default: ``0.5``.
            Returns:

            """
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect', anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]     
        # self.masks = self.masks.reshape(-1, 1, *self.input_size)
        # save_explanations('masks',self.masks*255)
        self.N = N
        self.p1 = p1



class DRISEExplainer(BlackBoxExplainer):
    def explain(self, data: np.ndarray) -> ExplainResult:
        """Explain the input."""
        raise NotImplementedError


class AutoExplainer(Explainer):
    """Explain in auto mode, using white box or black box approach."""

    def __init__(self, model: openvino.model_api.models.Model, explain_parameters: bool = None):
        super().__init__(model)
        self._explain_parameters = explain_parameters if explain_parameters else {}


class ClassificationAutoExplainer(AutoExplainer):
    """Explain classification models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray, target_explain_group: Optional[TargetExplainGroup] = None) -> ExplainResult:
        """
        Implements three explain scenarios, for different IR models:
            1. IR model contain xai branch -> infer Model API wrapper.
            2. If not (1), IR model can be augmented with XAI branch -> augment and infer.
            3. If not (1) and (2), IR model can NOT be augmented with XAI branch -> use XAI BB method.

        Args:
            data(numpy.ndarray): data to explain.
            target_explain_group(TargetExplainGroup): Target explain group.
        """
        if XAIModel.has_xai(self._model):
            logger.info("Model already has XAI - using White Box explainer.")
            explanations = WhiteBoxExplainer(self._model).explain(data, target_explain_group)
            return explanations
        else:
            try:
                logger.info("Model does not have XAI - trying to insert XAI and use White Box explainer.")
                self._model = XAIClassificationModel.insert_xai(self._model, self._explain_parameters)
                explanations = WhiteBoxExplainer(self._model).explain(data)
                return explanations
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model. Calling Black Box explainer.")
                explanations = RISEExplainer(self._model).explain(data)
                return explanations


class DetectionAutoExplainer(AutoExplainer):
    """Explain detection models in auto mode, using white box or black box approach."""

    def explain(self, data: np.ndarray) -> np.ndarray:
        """Explain the input."""
        raise NotImplementedError
