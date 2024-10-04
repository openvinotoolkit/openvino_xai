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
from openvino_xai.metrics.adcc import ADCC

MODEL_NAME = "mlc_mobilenetv3_large_voc"


class TestADCC:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.explainer = Explainer(
            model=self.model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )
        self.adcc = ADCC(self.model, self.preprocess_fn, self.postprocess_fn, self.explainer)

    def test_adcc(self):
        input_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        saliency_map = np.random.rand(224, 224)

        model_output = self.adcc.model_predict(input_image)
        class_idx = np.argmax(model_output)

        score = self.adcc(saliency_map, class_idx, input_image)
        complexity_score = score["complexity"]
        average_drop_score = score["average_drop"]
        coherency_score = score["coherency"]
        adcc_score = score["adcc"]
        assert complexity_score >= 0.2
        assert average_drop_score >= 0.2
        assert coherency_score >= 0.2
        assert adcc_score >= 0.4

        # Explainer(...embed_scaling=True)
        saliency_map = np.random.randint(0, 256, (224, 224))
        adcc_score = self.adcc(saliency_map, class_idx, input_image)["adcc"]
        assert 0 <= adcc_score <= 1

        # Explainer(...embed_scaling=False)
        saliency_map = np.random.uniform(-1, 1, (224, 224))
        adcc_score = self.adcc(saliency_map, class_idx, input_image)["adcc"]
        assert 0 <= adcc_score <= 1

    def test_evaluate(self):
        input_images = [np.random.rand(224, 224, 3) for _ in range(5)]
        explanations = [
            Explanation(
                {0: np.random.rand(224, 224), 1: np.random.rand(224, 224)}, targets=[0, 1], task=Task.CLASSIFICATION
            )
            for _ in range(5)
        ]

        adcc_score = self.adcc.evaluate(explanations, input_images)["adcc"]

        assert isinstance(adcc_score, float)
        assert 0 <= adcc_score <= 1

        # Activation map
        explanations = [
            Explanation({"per_image_map": np.random.rand(224, 224)}, targets="per_image_map", task=Task.CLASSIFICATION)
        ]
        adcc_score = self.adcc.evaluate(explanations, input_images)["adcc"]
        assert isinstance(adcc_score, float)
        assert 0 <= adcc_score <= 1
