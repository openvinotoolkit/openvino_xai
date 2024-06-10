# OpenVINO™ Explainable AI Toolkit Usage Guide

**OpenVINO™ Explainable AI (XAI) Toolkit** provides a suite of XAI algorithms for visual explanation of OpenVINO™ Intermediate Representation (IR) models.
Model explanation helps to identify the parts of the input that are responsible for the model's prediction,
which is useful for analyzing model's performance.

Current tutorial is primarily for classification CNNs.

OpenVINO XAI API documentation can be found [here](https://curly-couscous-ovjvm29.pages.github.io/).

Content:

- Explainer
- Basic usage: Auto mode
- White-Box mode
- Black-Box mode
- XAI insertion
- Example scripts


## Explainer - interface to XAI algorithms

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

Under the hood of the auto mode: will try to run white-box mode, if fails => will run black-box mode.
See more details about white_box and black-box modes below.

Generating saliency maps involves model inference. Explainer will perform model inference.
To infer, `preprocess_fn` and `postprocess_fn` are requested from the user, depending on the usage mode.

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

# Saving saliency maps
explanation.save("output_path", "name")
```


## White-Box mode

White-box mode is a two-step process that includes OV model update and further inference of the updated model.

Updated model has additional XAI branch inserted. XAI branch generates saliency maps during model inference. Saliency maps extend the list of model outputs, i.e. saliency maps are generated along with the original model outputs. Depending on the white-box algorithm, computational overhead of inserted XAI branch may vary, but it is usually relatively modest.

`preprocess_fn` is required to be provided by the user for the white-box mode.

```python
import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.explainer.parameters import ExplainMode, ExplanationParameters, TargetExplainGroup, VisualizationParameters
from openvino_xai.inserter.parameters import ClassificationInsertionParameters


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x


# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Optional - create insertion parameters
insertion_parameters = ClassificationInsertionParameters(
    # target_layer="last_conv_node_name",  # target_layer - node after which XAI branch will be inserted
    embed_scale=True,  # True by default.  If set to True, saliency map scale (0 ~ 255) operation is embedded in the model
    explain_method=xai.Method.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
)

# Explainer object will prepare and load the model once in the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
    explain_mode=ExplainMode.WHITEBOX,
    insertion_parameters=insertion_parameters,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")
voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
explanation_parameters = ExplanationParameters(
    target_explain_group=TargetExplainGroup.CUSTOM,
    target_explain_labels=[11, 14],  # target classes to explain, also ['dog', 'person'] is a valid input
    label_names=voc_labels,
    visualization_parameters=VisualizationParameters(overlay=True),  # by default, saliency map overlay over image
)
explanation = explainer(image, explanation_parameters)

# Saving saliency maps
explanation.save("output_path", "name")
```


## Black-Box mode

Black-box mode does not update the model (treating model as a black-box).
Black-box approaches are based on the perturbation of the input data and measurement of the model's output change.
The process is repeated many times, which requires hundreds or thousands of forward passes
and introduces significant computational overhead.

`preprocess_fn` and `postprocess_fn` are required to be provided by the user for the black-box mode.

```python
import cv2
import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

import openvino_xai as xai
from openvino_xai.explainer.explanation_parameters import ExplainMode, ExplanationParameters


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x


def postprocess_fn(x: OVDict):
    # Implementing own post-process function based on model's implementation
    # Output logits
    return x["logits"]


# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Explainer object will prepare and load the model once in the beginning
explainer = xai.Explainer(
    model,
    task=xai.Task.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
    postprocess_fn=postprocess_fn,
    explain_mode=ExplainMode.BLACKBOX,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")
explanation_parameters = ExplanationParameters(
    target_explain_labels=[11, 14],  # indices or string labels to explain
)
explanation = explainer(
    image,
    explanation_parameters,
    num_masks=1000,  # kwargs of the RISE algo
)

# Saving saliency maps
explanation.save("output_path", "name")
```


## XAI insertion (white-box usage)

As mentioned above, saliency map generation requires model inference.
In the above use cases, OVXAI performs model inference using provided processing functions.
Alternative approach is to use OVXAI just to insert XAI branch into the model and infer it in the original pipeline.

`insert_xai()` API is used for insertion.

Note: original model outputs are not affected and the model should be inferable by the original inference pipeline.

```python
import openvino.runtime as ov

import openvino_xai as xai
from openvino_xai.inserter.parameters import ClassificationInsertionParameters


# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Optional - create insertion parameters
insertion_parameters = ClassificationInsertionParameters(
    # target_layer="last_conv_node_name",  # target_layer - node after which XAI branch will be inserted
    embed_scale=True,  # True by default.  If set to True, saliency map scale (0 ~ 255) operation is embedded in the model
    explain_method=xai.Method.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
)

# Inserting XAI branch into the model graph
model_xai = xai.insert_xai(
    model=model,
    task=xai.Task.CLASSIFICATION,
    insertion_parameters=insertion_parameters,
)  # type: ov.Model

# ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****
```


## Example scripts

More usage scenarios are available in [examples](./../examples).

```python
# Retrieve models by running tests
# Models are downloaded and stored in .data/otx_models
pytest tests/test_classification.py

# Run a bunch of classification examples
# All outputs will be stored in the corresponding output directory
python examples/run_classification.py .data/otx_models/mlc_mobilenetv3_large_voc.xml \
tests/assets/cheetah_person.jpg --output output
```
