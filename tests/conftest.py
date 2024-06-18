# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

log = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser):
    """Add custom options for OpenVINO XAI tests."""
    parser.addoption(
        "--data-root",
        action="store",
        default=".data",
        help="Data root directory.",
    )
    parser.addoption(
        "--cache-root",
        action="store",
        default="~/.cache",
        help="Cache root directory.",
    )


@pytest.fixture(scope="session")
def fxt_data_root(request: pytest.FixtureRequest) -> Path:
    """Data root directory path."""
    data_root = Path(request.config.getoption("--data-root")).expanduser().absolute()
    msg = f"{data_root = }"
    log.info(msg)
    return data_root


@pytest.fixture(scope="session", autouse=True)
def fxt_cache_root(request: pytest.FixtureRequest) -> Path:
    """Cache root directory path."""
    cache_root = Path(request.config.getoption("--cache-root")).expanduser().absolute()
    os.environ["XDG_CACHE_HOME"] = cache_root.as_posix()
    os.environ["HF_HOME"] = cache_root.as_posix()
    msg = f"{cache_root = }"
    print(msg)
    log.info(msg)
    return cache_root
