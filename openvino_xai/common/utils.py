# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Common functionality.
"""

import os
from pathlib import Path

import logging
from urllib.request import urlretrieve

import openvino.runtime as ov


logger = logging.getLogger("openvino_xai")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


SALIENCY_MAP_OUTPUT_NAME = "saliency_map"


def has_xai(model: ov.Model) -> bool:
    """
    Function checks if the model contains XAI branch.

    :param model: OV IR model.
    :type model: openvino.runtime.Model
    :return: True is the model has XAI branch and saliency_map output, False otherwise.
    """
    if not isinstance(model, ov.Model):
        raise ValueError(f"Input model has to be openvino.runtime.Model instance, but got{type(model)}.")
    for output in model.outputs:
        if SALIENCY_MAP_OUTPUT_NAME in output.get_names():
            return True
    return False


def retrieve_otx_model(data_dir, model_name, dir_url=None):
    destination_folder = Path(data_dir) / "otx_models"
    os.makedirs(destination_folder, exist_ok=True)
    if dir_url is None:
        dir_url = f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}"
        snapshot_file = "openvino"
    else:
        snapshot_file = model_name

    for post_fix in ["xml", "bin"]:
        if not os.path.isfile(os.path.join(destination_folder, model_name + f".{post_fix}")):
            urlretrieve(
                f"{dir_url}/{snapshot_file}.{post_fix}",
                f"{destination_folder}/{model_name}.{post_fix}",
            )
