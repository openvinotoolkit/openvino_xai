# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import logging
from urllib.request import urlretrieve

logger = logging.getLogger("openvino_xai")
logger.setLevel(logging.INFO)


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
