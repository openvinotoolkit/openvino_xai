# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import openvino.runtime as ov

from openvino_xai.common.parameters import Task
from openvino_xai.common.utils import has_xai, retrieve_otx_model
from openvino_xai.inserter.inserter import insert_xai
from tests.integration.test_classification import DEFAULT_CLS_MODEL

DARA_DIR = Path(".data")


def test_has_xai():
    model_without_xai = DEFAULT_CLS_MODEL
    retrieve_otx_model(DARA_DIR, model_without_xai)
    model_path = DARA_DIR / "otx_models" / (model_without_xai + ".xml")
    model = ov.Core().read_model(model_path)

    assert not has_xai(model)

    model_xai = insert_xai(
        model,
        task=Task.CLASSIFICATION,
    )

    assert has_xai(model_xai)
