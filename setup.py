# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIR = Path(__file__).resolve().parent

setup(
    name="openvino_xai",
    version="0.0.1",
    description="OV XAI: a toolbox for explaining models in OpenVINO fromat",
    author="Intel(R) Corporation",
    url="https://github.com/intel-sandbox/openvino_xai",
    packages=find_packages(SETUP_DIR),
    setup_requires=['wheel'],
    package_dir={"openvino_xai": str(SETUP_DIR / "openvino_xai")},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    license="OSI Approved :: Apache Software License",
    python_requires=">=3.7",
    install_requires=(SETUP_DIR / "requirements" / "base.txt").read_text(),
    extras_require={
        "dev": (SETUP_DIR / "requirements" / "dev.txt").read_text(),
    },
    long_description=(SETUP_DIR / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
