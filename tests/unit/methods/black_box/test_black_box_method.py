# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
import pytest

from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.black_box_methods import RISE
from tests.integration.test_classification import DEFAULT_CLS_MODEL


class TestRISE:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    data_dir = Path(".data")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn()

    @pytest.mark.parametrize("explain_target_indices", [[0], None])
    def test_run(self, explain_target_indices):
        retrieve_otx_model(self.data_dir, DEFAULT_CLS_MODEL)
        model_path = self.data_dir / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        model = ov.Core().read_model(model_path)
        compiled_model = ov.Core().compile_model(model, "CPU")

        saliency_map = RISE.run(
            compiled_model,
            self.preprocess_fn,
            self.postprocess_fn,
            self.image,
            explain_target_indices,
            num_masks=5,
        )
        assert saliency_map.dtype == np.uint8
        assert saliency_map.shape == (1, 20, 224, 224)
        assert (saliency_map >= 0).all() and (saliency_map <= 255).all()
