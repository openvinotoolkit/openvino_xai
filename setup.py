# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIR = Path(__file__).resolve().parent

INSTALL_REQUIRES = (SETUP_DIR / "requirements" / "base.txt").read_text()

TEST_BASE_EXTRAS = (SETUP_DIR / "requirements" / "dev.txt").read_text()
TIMM_EXTRAS = TEST_BASE_EXTRAS + "\n" + (SETUP_DIR / "requirements" / "dev_timm.txt").read_text()
EXTRAS_REQUIRE = {
    "dev": TEST_BASE_EXTRAS,
    "dev_timm": TIMM_EXTRAS,
}

setup(
    name="openvino_xai",
    version="0.0.1",
    description="OV XAI: a toolbox for explaining models in OpenVINO format",
    author="Intel(R) Corporation",
    url="https://github.com/intel-sandbox/openvino_xai",
    packages=find_packages(SETUP_DIR),
    setup_requires=["wheel"],
    package_dir={"openvino_xai": str(SETUP_DIR / "openvino_xai")},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    license="OSI Approved :: Apache Software License",
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    long_description=(SETUP_DIR / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
