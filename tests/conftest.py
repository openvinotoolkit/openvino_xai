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
        "--output-root",
        action="store",
        default=".data",
        help="Output root directory.",
    )
    parser.addoption(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Whether to delete model cahce directory. Defaults to False.",
    )


@pytest.fixture(scope="session")
def fxt_data_root(request: pytest.FixtureRequest) -> Path:
    """Data root directory path."""
    data_root = Path(request.config.getoption("--data-root"))
    msg = f"{data_root = }"
    log.info(msg)
    print(msg)
    return data_root


@pytest.fixture(scope="session")
def fxt_output_root(request: pytest.FixtureRequest) -> Path:
    """Output root directory path."""
    output_root = Path(request.config.getoption("--output-root"))
    output_root.mkdir(parents=True, exist_ok=True)
    msg = f"{output_root = }"
    log.info(msg)
    print(msg)
    return output_root


@pytest.fixture(scope="session")
def fxt_clear_cache(request: pytest.FixtureRequest) -> Path:
    """Data root directory path."""
    clear_cache = bool(request.config.getoption("--clear-cache"))
    msg = f"{clear_cache = }"
    log.info(msg)
    print(msg)
    return clear_cache
