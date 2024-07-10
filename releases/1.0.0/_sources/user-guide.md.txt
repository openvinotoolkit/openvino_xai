# OpenVINO™ Explainable AI Toolkit User Guide

**OpenVINO™ Explainable AI (XAI) Toolkit** provides a suite of XAI algorithms for visual explanation of OpenVINO™ Intermediate Representation (IR) models.
Model explanation helps to identify the parts of the input that are responsible for the model's prediction,
which is useful for analyzing model's performance.

Current tutorial is primarily for classification CNNs.

OpenVINO XAI API documentation can be found [here](https://openvinotoolkit.github.io/openvino_xai/).

Content:

- [OpenVINO™ Explainable AI Toolkit User Guide](#openvino-explainable-ai-toolkit-user-guide)
  - [OpenVINO XAI Architecture](#openvino-xai-architecture)
  - [Explainer - interface to XAI algorithms](#explainer---interface-to-xai-algorithms)
  - [Basic usage: Auto mode](#basic-usage-auto-mode)
    - [Running without `preprocess_fn`](#running-without-preprocess_fn)
    - [Specifying `preprocess_fn`](#specifying-preprocess_fn)
  - [White-Box mode](#white-box-mode)
  - [Black-Box mode](#black-box-mode)
  - [XAI insertion (white-box usage)](#xai-insertion-white-box-usage)
  - [Example scripts](#example-scripts)


## OpenVINO XAI Architecture

![OpenVINO XAI Architecture](_static/ovxai-architecture.svg)

OpenVINO XAI provides the API to explain models, using two types of methods:

- **White-box** - treats the model as a white box, making inner modifications and adding an extra XAI branch. This results in additional output from the model and relatively fast explanations.
- **Black-box** - treats the model as a black box, working on a wide range of models. However, it requires many more inference runs.

## Explainer - interface to XAI algorithms

In a nutshell, the explanation call looks like this:

```python
import openvino_xai as xai

explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)
explanation = explainer(data, explanation_parameters)
```


## Basic usage: Auto mode

The easiest way to run the explainer is in Auto mode. Under the hood, Auto mode will try to run in `White-Box` mode first. If it fails, it will run in `Black-Box` mode.

Learn more details about [White-Box](#white-box-mode) and [Black-Box](#black-box-mode) modes below.

![Auto mode process](_static/auto_explain_mode.jpg)

Generating saliency maps involves model inference. Explainer will perform model inference but to do it, it requires `preprocess_fn` and `postprocess_fn`.

### Running without `preprocess_fn`

Here's the example how we can avoid passing `preprocess_fn` by preprocessing data beforehand (like resizing and adding a batch dimension).

```python
import cv2
import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

import openvino_xai as xai


def postprocess_fn(x: OVDict):
    # Implementing our own post-process function based on the model's implementation
    # Return "logits" model output
    return x["logits"]

# Create ov.Model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Explainer object will prepare and load the model once in the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    postprocess_fn=postprocess_fn,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")

# Pre-process the image as the model requires (resizing and adding a batch dimension)
preprocessed_image = cv2.resize(src=image, dsize=(224, 224))
preprocessed_image = np.expand_dims(preprocessed_image, 0)

# Run explanation
explanation = explainer(
    preprocessed_image,
    target_explain_labels=[11, 14],  # indices or string labels to explain
    overlay=True,  # False by default
    original_input_image=image,  # to apply overlay on the original image instead of the preprocessed one that was used for the explainer
)

# Save saliency maps
explanation.save("output_path", "name")
```

### Specifying `preprocess_fn`


```python
import cv2
import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

import openvino_xai as xai


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing our own pre-process function based on the model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x

def postprocess_fn(x: OVDict):
    # Implementing our own post-process function based on the model's implementation
    # Return "logits" model output
    return x["logits"]

# Create ov.Model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# The Explainer object will prepare and load the model once in the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
    postprocess_fn=postprocess_fn,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")

# Run explanation
explanation = explainer(
    image,
    target_explain_labels=[11, 14],  # indices or string labels to explain
)

# Save saliency maps
explanation.save("output_path", "name")
```


## White-Box mode

White-box mode involves two steps: updating the OV model and then running the updated model.

The updated model has an extra XAI branch resulting in an additional `saliency_map` output. This XAI branch creates saliency maps during the model's inference. The computational load from the XAI branch varies depending on the white-box algorithm, but it's usually quite small.

You need to pass either `preprocess_fn` or already preprocessed images to run the explainer in white-box mode.

```python
import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.explainer import ExplainMode


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on the model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x

# Create ov.Model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# The Explainer object will prepare and load the model once at the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")

voc_labels = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Run explanation
explanation = explainer(
    image,
    explain_mode=ExplainMode.WHITEBOX,
    # target_layer="last_conv_node_name",  # target_layer - node after which the XAI branch will be inserted, usually the last convolutional layer in the backbone
    embed_scaling=True,  # True by default. If set to True, the saliency map scale (0 ~ 255) operation is embedded in the model
    explain_method=xai.Method.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
    label_names=voc_labels,
    target_explain_labels=[11, 14],  # target classes to explain, also ['dog', 'person'] is a valid input, since label_names are provided
    overlay=True,  # False by default 
)

# Save saliency maps
explanation.save("output_path", "name")
```


## Black-Box mode

Black-box mode does not update the model (treating the model as a black box).
Black-box approaches are based on the perturbation of the input data and measurement of the model's output change.

The process is repeated many times, requiring hundreds or thousands of forward passes and introducing **significant computational overhead**.

`preprocess_fn` (or preprocessed images) and `postprocess_fn` are required to be provided by the user for black-box mode.

```python
import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.explainer import ExplainMode


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing our own pre-process function based on the model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x

# Create ov.Model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# The Explainer object will prepare and load the model once at the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")

# Run explanation
explanation = explainer(
    image,
    explain_mode=ExplainMode.BLACKBOX,
    target_explain_labels=[11, 14],  # target classes to explain
    # target_explain_labels=-1,  # explain all classes
    overlay=True,  # False by default 
    num_masks=1000,  # kwargs for the RISE algorithm
)

# Save saliency maps
explanation.save("output_path", "name")

```


## XAI insertion (white-box usage)

As mentioned above, saliency map generation requires model inference. In the above use cases, OpenVINO XAI performs model inference using provided processing functions. An alternative approach is to use XAI to insert the XAI branch into the model and infer it in the original pipeline.

`insert_xai()` API is used for insertion.

**Note**: The original model outputs are not affected, and the model should be inferable by the original inference pipeline.

```python
import openvino.runtime as ov
import openvino_xai as xai


# Create an ov.Model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Insert XAI branch into the model graph
model_xai = xai.insert_xai(
    model=model,
    task=xai.Task.CLASSIFICATION,
    # target_layer="last_conv_node_name",  # target_layer - the node after which the XAI branch will be inserted, usually the last convolutional layer in the backbone
    embed_scaling=True,  # True by default. If set to True, the saliency map scale (0 ~ 255) operation is embedded in the model
    explain_method=xai.Method.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
)  # type: ov.Model

# ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****
```


## Example scripts

More usage scenarios that can be used with your own models and images as arguments are available in [examples](../../examples).

```python
# Retrieve models by running tests
# Models are downloaded and stored in .data/otx_models
pytest tests/test_classification.py

# Run a bunch of classification examples
# All outputs will be stored in the corresponding output directory
python examples/run_classification.py .data/otx_models/mlc_mobilenetv3_large_voc.xml 
tests/assets/cheetah_person.jpg --output output
```
