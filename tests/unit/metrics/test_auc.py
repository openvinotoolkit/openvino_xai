import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.explanation import Explanation
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
)
from openvino_xai.metrics.insertion_deletion_auc import InsertionDeletionAUC

MODEL_NAME = "mlc_mobilenetv3_large_voc"


class TestAUC:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)
    steps = 10

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        core = ov.Core()
        model = core.read_model(model_path)
        self.auc = InsertionDeletionAUC(model, self.preprocess_fn, self.postprocess_fn)

        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

    def test_insertion_deletion_auc(self):
        class_idx = 1
        input_image = np.random.rand(224, 224, 3)
        saliency_map = np.random.rand(224, 224)

        insertion_auc, deletion_auc = self.auc(saliency_map, class_idx, input_image, self.steps).values()

        for value in [insertion_auc, deletion_auc]:
            assert isinstance(value, float)
            assert 0 <= value <= 1

    def test_evaluate(self):
        input_images = [np.random.rand(224, 224, 3) for _ in range(5)]
        explanations = [
            Explanation(
                {0: np.random.rand(224, 224), 1: np.random.rand(224, 224)}, targets=[0, 1], task=Task.CLASSIFICATION
            )
            for _ in range(5)
        ]

        insertion, deletion, delta = self.auc.evaluate(explanations, input_images, self.steps).values()
        for value in [insertion, deletion]:
            assert isinstance(value, float)
            assert 0 <= value <= 1
        assert isinstance(delta, float)

        # Activation map
        explanations = [
            Explanation({"per_image_map": np.random.rand(224, 224)}, targets="per_image_map", task=Task.CLASSIFICATION)
        ]
        insertion, deletion, delta = self.auc.evaluate(explanations, input_images, self.steps).values()
        for value in [insertion, deletion]:
            assert isinstance(value, float)
            assert 0 <= value <= 1
        assert isinstance(delta, float)
