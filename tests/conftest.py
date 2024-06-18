# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import logging
from pathlib import Path


log = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser):
    """Add custom options for OpenVINO XAI tests."""
    parser.addoption(
        "--data-root",
        action="store",
        default=".data",
        help="Data root directory.",
    )


@pytest.fixture(scope="session")
def fxt_data_root(request: pytest.FixtureRequest) -> Path:
    """Data root directory path."""
    data_root = Path(request.config.getoption("--data-root"))
    msg = f"{data_root = }"
    log.info(msg)
    return data_root
