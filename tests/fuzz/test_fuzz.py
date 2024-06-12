# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import atheris

with atheris.instrument_imports():
    from openvino_xai import Explainer, insert_xai


def TestOneInput(input_bytes):
    data_provider = atheris.FuzzedDataProvider(input_bytes)

    # Test Explainer statefull class API
    try:
        explainer = Explainer(
            model=data_provider.ConsumeBytes(8),
            task=data_provider.ConsumeUInt(8),
        )
        explanation = explainer(data=data_provider.ConsumeBytes(8))
    except ValueError as e:
        pass

    # Test insert_xai() stateless API
    try:
        insert_xai(
            model=data_provider.ConsumeBytes(8),
            task=data_provider.ConsumeUInt(8),
        )
    except ValueError as e:
        pass


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
