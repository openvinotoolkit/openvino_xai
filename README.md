# OpenVINO™ Explainable AI Toolkit

**OpenVINO™ Explainable AI (XAI) Toolkit** provides a suite of XAI algorithms for visual explanation of
[OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

## Documentation

OpenVINO XAI API documentation can be found [here](https://curly-couscous-ovjvm29.pages.github.io/).

## Installation

- Set up an isolated python environment:

```bash
# Create virtual env.
python3 -m venv .ovxai

# Activate virtual env.
source .ovxai/bin/activate
```

- Package installation:

```bash
# Package mode (for normal use):
pip install .

# Editable mode (for development):
pip install -e .[dev]
```

- Verification:

```bash
# Run tests
pytest -v -s ./tests/

# Run code quality checks
pre-commit run --all-files
```


## Usage

To explain [OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) you only need
preprocessing function (and sometimes postprocessing).

```python
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)
explanation = explainer(data, explanation_parameters)
```

By default the model will be explained using `auto mode`.
Under the hood of the `auto mode`: will try to run `white-box mode`, if fails => will run `black-box mode`.

Generating saliency maps involves model inference. Explainer will perform model inference.
To infer, `preprocess_fn` and `postprocess_fn` are requested from the user.
`preprocess_fn` is always required, `postprocess_fn` is required only for black-box.

```python
import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.explainer.explanation_parameters import ExplanationParameters


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x


# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Explainer object will prepare and load the model once in the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")
explanation_parameters = ExplanationParameters(
    target_explain_labels=[11, 14],  # indices or string labels to explain
)
explanation = explainer(image, explanation_parameters)

explanation: Explanation
explanation.saliency_map: Dict[int: np.ndarray]  # key - class id, value - processed saliency map e.g. 354x500x3

# Saving saliency maps
explanation.save("output_path", "name")
```

See more usage scenarios in [Usage.md](.docs/Usage.md) and [examples](./examples).

### Running example scripts

```python
# Retrieve OTX models by running tests
# Models are downloaded and stored in .data/otx_models
pytest tests/test_classification.py

# Run a bunch of classification examples
# All outputs will be stored in the corresponding output directory
python examples/run_classification.py .data/otx_models/mlc_mobilenetv3_large_voc.xml \
tests/assets/cheetah_person.jpg --output output
```

## Scope of explained models

Models from [Pytorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) are used
for classification benchmark.

### White-box (fast, model-dependent)

#### Classification

We benchmarked white-box explanation (using ReciproCAM explain method) using 528 models.
Currently, we support only CNN-based architectures in white-box mode,
transformers will be supported in the upcoming weeks.

For more details (statistic, model list, samples of generated saliency maps) see
[#11](https://github.com/intel-sandbox/openvino_xai/pull/11).

### Black-box (slow, model-agnostic)

#### Classification

We benchmarked black-box explanation (using RISE explain method) using 528 CNN models and 115 transformer-based models.
Black-box explainer support all types of models that output logits (e.g. CNNs, transformers, etc.).

For more details (statistic, model list, samples of generated saliency maps) see
[#23](https://github.com/intel-sandbox/openvino_xai/pull/23).
