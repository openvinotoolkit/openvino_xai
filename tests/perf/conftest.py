# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from cpuinfo import get_cpu_info

log = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser):
    """Add custom options for OpenVINO XAI perf tests."""
    parser.addoption(
        "--num-repeat",
        action="store",
        default=5,
        help="Number of trials for each model explain. "
        "Random seeds are set to 0 ~ num_repeat-1 for the trials. "
        "Defaults to 10.",
    )
    parser.addoption(
        "--num-masks",
        action="store",
        default=5000,
        help="Number of masks for black box methods." "Defaults to 5000.",
    )


@pytest.fixture(scope="session")
def fxt_num_repeat(request: pytest.FixtureRequest) -> int:
    """Number of repeated trials."""
    num_repeat = int(request.config.getoption("--num-repeat"))
    msg = f"{num_repeat = }"
    log.info(msg)
    print(msg)
    return num_repeat


@pytest.fixture(scope="session")
def fxt_num_masks(request: pytest.FixtureRequest) -> int:
    """Number of masks for black box methods."""
    num_masks = int(request.config.getoption("--num-masks"))
    msg = f"{num_masks = }"
    log.info(msg)
    print(msg)
    return num_masks


@pytest.fixture(scope="session")
def fxt_current_date() -> str:
    tz = timezone(offset=timedelta(hours=9), name="Seoul")
    return datetime.now(tz=tz).strftime("%Y%m%d-%H%M%S")


@pytest.fixture(scope="session")
def fxt_output_root(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    fxt_current_date: str,
) -> Path:
    """Output root + dateh."""
    output_root = request.config.getoption("--output-root")
    if output_root is None:
        output_root = tmp_path_factory.mktemp("openvino_xai")
    output_root = Path(output_root) / "perf" / fxt_current_date
    output_root.mkdir(parents=True, exist_ok=True)
    msg = f"{output_root = }"
    log.info(msg)
    print(msg)
    return output_root


@pytest.fixture(scope="session")
def fxt_tags(fxt_current_date: str) -> dict[str, str]:
    """Tag fields to record various metadata."""
    try:
        from importlib.metadata import version

        version_str = version("openvino_xai")
    except Exception:
        version_str = "unknown"
    try:
        branch_str = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        )  # noqa: S603, S607
    except Exception:
        branch_str = os.environ.get("GH_CTX_REF_NAME", "unknown")
    try:
        commit_str = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        )  # noqa: S603, S607
    except Exception:
        commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    tags = {
        "version": version_str,
        "branch": branch_str,
        "commit": commit_str,
        "date": fxt_current_date,
        "machine_name": platform.node(),
        "cpu_info": get_cpu_info()["brand_raw"],
    }
    msg = f"{tags = }"
    log.info(msg)
    return tags


@pytest.fixture(scope="session", autouse=True)
def fxt_perf_summary(
    fxt_output_root: Path,
    fxt_tags: dict[str, str],
):
    """Summarize all results at the end of test session."""
    yield

    # Merge all raw data
    raw_data = []
    csv_files = fxt_output_root.rglob("perf-raw-*-*.csv")
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        raw_data.append(data)
    if len(raw_data) == 0:
        print("No raw data to summarize")
        return
    raw_data = pd.concat(raw_data, ignore_index=True)
    raw_data = raw_data.drop(["Unnamed: 0"], axis=1)
    raw_data = raw_data.replace(
        {
            "Method.RECIPROCAM": "RECIPROCAM",
            "Method.VITRECIPROCAM": "RECIPROCAM",
            "Method.RISE": "RISE",
        }
    )
    raw_data.to_csv(fxt_output_root / "perf-raw-all.csv", index=False)

    # Summarize
    data = raw_data.pivot_table(
        index=["model", "version"],
        columns=["method"],
        values=["time"],
        aggfunc=["mean", "std"],
    )
    data.columns = data.columns.rename(["stat", "metric", "method"])
    data = data.reorder_levels(["method", "metric", "stat"], axis=1)
    data0 = data

    data = raw_data.pivot_table(
        index=["version"],
        columns=["method"],
        values=["time"],
        aggfunc=["mean", "std"],
    )
    indices = data.index.to_frame()
    indices["model"] = "all"
    data.index = pd.MultiIndex.from_frame(indices)
    data = data.reorder_levels(["model", "version"], axis=0)
    data.columns = data.columns.rename(["stat", "metric", "method"])
    data = data.reorder_levels(["method", "metric", "stat"], axis=1)
    data1 = data

    data = pd.concat([data0, data1], axis=0)
    data = data.sort_index(axis=0).sort_index(axis=1)

    print("=" * 20, "[Perf summary]")
    print(data)
    data.to_csv(fxt_output_root / "perf-summary.csv")
    data.to_excel(fxt_output_root / "perf-summary.xlsx")
    print(f"    -> Saved to {fxt_output_root}")
