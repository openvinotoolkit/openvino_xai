<div align="center">

# OpenVINO™ Explainable AI Toolkit - OpenVINO XAI

---

[Features](#features) •
[Install](#installation) •
[Quick Start](#quick-start) •
[License](#license) •
[Documentation](https://openvinotoolkit.github.io/openvino_xai/releases/1.1.0)

![Python](https://img.shields.io/badge/python-3.10%2B-green)
[![OpenVINO](https://img.shields.io/badge/openvino-2024.4-purple)](https://pypi.org/project/openvino/)
[![codecov](https://codecov.io/gh/openvinotoolkit/openvino_xai/graph/badge.svg?token=NR0Z0CWDK9)](https://codecov.io/gh/openvinotoolkit/openvino_xai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/openvino_xai)](https://pypi.org/project/openvino_xai)
[![Downloads](https://static.pepy.tech/badge/openvino_xai)](https://pepy.tech/project/openvino_xai)

---

</div>

![OpenVINO XAI Concept](docs/source/_static/ovxai-concept.svg)

**OpenVINO™ Explainable AI (XAI) Toolkit** provides a suite of XAI algorithms for visual explanation of
[**OpenVINO™**](https://github.com/openvinotoolkit/openvino) as well as [**PyTorch**](https://pytorch.org) and [**ONNX**](https://onnx.ai/) models.

Given AI models and input images, **OpenVINO XAI** generates **saliency maps**
which highlights regions of the interest in the inputs from the models' perspective
to help users understand the reason why the complex AI models output such responses.

Using this package, you can augment the model analysis & explanation feature
on top of the existing OpenVINO inference pipeline with a few lines of code.

```python
import openvino_xai as xai

explainer = xai.Explainer(model=ov_model, task=xai.Task.CLASSIFICATION)

# Existing inference pipeline
for i, image in enumerate(images):
    labels = infer(model=ov_model, image=image)

    # Model analysis
    explanation = explainer(data=image, targets=labels)
    explanation.save(dir_path="./xai", name=str(i))
```

---

## Features

### What's new in v1.1.0

* Support PyTorch models with `insert_xai()` API for saliency map generation on PyTorch / ONNX runtime
* Support OpenVINO IR (.xml) / ONNX (.onnx) model files for `Explainer`
* Enable AISE method: Adaptive Input Sampling for Explanation of Black-box Models
* Add Pointing Game, Insertion-Deletion AUC and ADCC quality metrics for saliency maps
* Upgrade OpenVINO to 2024.4.0
* Add saliency map visualization with explanation.plot()
* Enable flexible naming for saved saliency maps and include confidence scores

Please refer to the [change logs](CHANGELOG.md) for the full release history.

### Supported XAI methods

At the moment, *Image Classification* and *Object Detection* tasks are supported for the *Computer Vision* domain.
*Black-Box* (model agnostic but slow) methods and *White-Box* (model specific but fast) methods are supported:

| Domain          | Task                 | Type      | Algorithm              | Links |
|-----------------|----------------------|-----------|------------------------|-------|
| Computer Vision | Image Classification | White-Box | ReciproCAM             | [arxiv](https://arxiv.org/abs/2209.14074) / [src](openvino_xai/methods/white_box/recipro_cam.py) |
|                 |                      |           | VITReciproCAM          | [arxiv](https://arxiv.org/abs/2310.02588) / [src](openvino_xai/methods/white_box/recipro_cam.py) |
|                 |                      |           | ActivationMap          | experimental / [src](openvino_xai/methods/white_box/activation_map.py)                           |
|                 |                      | Black-Box | AISEClassification     | [src](openvino_xai/methods/black_box/aise/classification.py)                                     |
|                 |                      |           | RISE                   | [arxiv](https://arxiv.org/abs/1806.07421v3) / [src](openvino_xai/methods/black_box/rise.py)      |
|                 | Object Detection     | White-Box | DetClassProbabilityMap | experimental / [src](openvino_xai/methods/white_box/det_class_probability_map.py)                |
|                 |                      | Black-Box | AISEDetection          | [src](openvino_xai/methods/black_box/aise/detection.py)                                          |

See more method comparison at the [User Guide](docs/source/user-guide.md).

### Supported explainable models

Most of CNNs and Transformer models from [Pytorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) are supported and validated.

Please refer to the following known issues for unsupported models and reasons.

* [Runtime error from ONNX / OpenVINO IR models while conversion or inference for XAI (#29)](https://github.com/openvinotoolkit/openvino_xai/issues/29)
* [Models not supported by white box XAI methods (#30)](https://github.com/openvinotoolkit/openvino_xai/issues/30)

> **_NOTE:_**  GenAI / LLMs would be also supported incrementally in the upcoming releases.

---

## Installation

> **_NOTE:_**  OpenVINO XAI works on Python 3.10 or higher

<details>
<summary>Set up environment</summary>

```bash
# Create virtual env.
python3.10 -m venv .ovxai

# Activate virtual env.
source .ovxai/bin/activate
```
</details>

Install from PyPI package

```bash
# Base package (for normal use):
pip install openvino_xai

# Dev package (for development):
pip install openvino_xai[dev]
```

<details>
<summary>Install from source</summary>

```bash
# Clone the source repository
git clone https://github.com/openvinotoolkit/openvino_xai.git
cd openvino_xai

# Editable mode (for development):
pip install -e .[dev]
```
</details>

<details>
<summary>(Optional) Enable PyTorch support</summary>

You can enjoy the PyTorch XAI feature if the PyTorch is installed along with the OpenVINO XAI.

```bash
# Install PyTorch (CPU version as example)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```
Please refer to the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for other options.
</details>

<details>
<summary>Verify installation</summary>

```bash
# Run tests
pytest -v -s ./tests/unit

# Run code quality checks
pre-commit run --all-files
```
</details>

---

## Quick Start

### Hello, OpenVINO XAI

Let's imagine the case that our OpenVINO model is up and running on a inference pipeline.
While watching the outputs, we may want to analyze the model's behavior for debugging or understanding purposes.

By using the **OpenVINO XAI** `Explainer`, we can visualize why the model gives such responses.
In this example, we are trying to know the reason why the model outputs a `cheetah` label for the given input image.

```python
import cv2
import numpy as np
import openvino as ov
import openvino_xai as xai

# Load the model: IR or ONNX
ov_model: ov.Model = ov.Core().read_model("mobilenet_v3.xml")

# Load the image to be analyzed
image: np.ndarray = cv2.imread("tests/assets/cheetah_person.jpg")
image = cv2.resize(image, dsize=(224, 224))
image = np.expand_dims(image, 0)

# Create the Explainer for the model
explainer = xai.Explainer(
    model=ov_model,  # accepts path arguments "mobilenet_v3.xml" or "mobilenet_v3.onnx" as well
    task=xai.Task.CLASSIFICATION,
)

# Generate saliency map for the label of interest
explanation: xai.Explanation = explainer(
    data=image,
    targets=293,  # (cheetah), accepts label indices or actual label names if label_names provided
    overlay=True,  # saliency map overlay over the input image, defaults to False
)

# Save saliency maps to output directory
explanation.save(dir_path="./output")
```

Original image | Explained image
---------------|----------------
![Oringinal images](tests/assets/cheetah_person.jpg) | ![Explained image](docs/source/_static/xai-cheetah.png)

We can see that model is focusing on the body or skin area of the animals to tell if this image contains actual cheetahs.

### Insert XAI head to your models

Using the `insert_xai` API, we can insert XAI head to existing OpenVINO or PyTorch models directly and get additional "saliency_map" output in the same inference pipeline.

```python
import torch
import timm

# Get a PyTorch model from TIMM
torch_model: torch.nn.Module = timm.create_model("resnet18.a1_in1k", in_chans=3, pretrained=True)

# Insert XAI head
model_xai: torch.nn.Module = xai.insert_xai(torch_model, xai.Task.CLASSIFICATION)

# Torch XAI model inference
model_xai.eval()
with torch.no_grad():
    outputs = model_xai(torch.from_numpy(image_norm))
    logits = outputs["prediction"]  # BxC
    saliency_maps = outputs["saliency_map"]  # BxCxHxW: per-class saliency map
```

### More advanced use-cases

Users could tweak the basic use-case according to their purpose, which include but not limited to:

* Select XAI mode (White-Box or Black-Box) or even specific method which are automatically decided by default
* Provide custom model pre/post processing functions like resize and normalizations which the model expects
* Customize output image visualization options
* Explain multiple class targets, passing them as label indices or as actual label names
* Call explainer multiple times to explain multiple images or to use different targets
* Insert XAI head to your PyTorch models and export to ONNX format to generate saliency maps on ONNX Runtime
  (Refer to the [full example script](./examples/run_torch_onnx.py))

Please find more options and scenarios in the following links:

* [OpenVINO XAI User Guide](docs/source/user-guide.md)
* [OpenVINO Notebook - XAI Basic](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/explainable-ai-1-basic/explainable-ai-1-basic.ipynb)
* [OpenVINO Notebook - XAI Deep Dive](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/explainable-ai-2-deep-dive/explainable-ai-2-deep-dive.ipynb)
* [OpenVINO Notebook - Saliency Map Interpretation](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/explainable-ai-3-map-interpretation/explainable-ai-3-map-interpretation.ipynb)

### Playing with the examples

Please look around the runnable [example scripts](./examples) and play with them to get used to the `Explainer` and `insert_xai` APIs.

```bash
# Prepare models by running tests (need "pip install openvino_xai[dev]" extra option)
# Models are downloaded and stored in .data/otx_models
pytest tests/test_classification.py

# Run a bunch of classification examples
# All outputs will be stored in the corresponding output directory
python examples/run_classification.py .data/otx_models/mlc_mobilenetv3_large_voc.xml \
tests/assets/cheetah_person.jpg --output output

# Run PyTorch and ONNX support example
python examples/run_torch_onnx.py
```

---

## Contributing

For those who would like to contribute to the library, please refer to the [contribution guide](CONTRIBUTING.md) for details.

Please let us know via the [Issues tab](https://github.com/openvinotoolkit/openvino_xai/issues/new) if you have any issues, feature requests, or questions.

Thank you! We appreciate your support!

<a href="https://github.com/openvinotoolkit/openvino_xai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/openvino_xai" />
</a>

---

## License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---

## Disclaimer

Intel is committed to respecting human rights and avoiding complicity in human rights abuses.
See Intel's [Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html).
Intel's products and software are intended only to be used in applications that do not cause or contribute to a violation of an internationally recognized human right.

---
