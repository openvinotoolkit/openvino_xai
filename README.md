# OpenVINO-XAI

OpenVINO-XAI provides a suite of eXplainable AI (XAI) algorithms for explanation of
[OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR).

# Documentation

OpenVINO-XAI API documentation can be found [here](https://curly-couscous-ovjvm29.pages.github.io/).

# Installation

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
pre-commit run -a
```

# Usage in white-box mode

## Insertion: insert XAI branch into the model

```python
import openvino.runtime as ov
import openvino_xai as ovxai
from openvino_xai.common.parameters import TaskType

# Creating original model
model: ov.Model
model = ov.Core().read_model("path/to/model.xml")

# Inserting XAI branch into the model graph
model_xai: ov.Model
model_xai = ovxai.insert_xai(model, task_type=TaskType.CLASSIFICATION)

# ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****
```

## Explanation: generate explanation via inference

### Get raw saliency maps: use original model inference pipeline

```python
# Compile model with XAI branch
compiled_model = ov.Core().compile_model(model_xai, "CPU")

# User's code that creates a callable model_inferrer
def model_inferrer(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Custom inference pipeline
    image_processed = preprocess(image)
    result = compiled_model([image_processed])
    logits = postprocess(result["logits"])
    raw_saliency_map = result["saliency_map"]  # "saliency_map" is an additional model output added during insertion
    return logits, raw_saliency_map

# Get raw saliency map via model_inferrer call
logits, raw_saliency_map = model_inferrer(cv2.imread("path/to/image.jpg"))
raw_saliency_map: np.ndarray  # e.g. 20x7x7 uint8 array
```

### Get processed saliency maps: 1. modify output format of the original model inference pipeline and 2. call .explain()

```python
from openvino_xai.explanation.utils import InferenceResult
from openvino_xai.explanation.explanation_parameters import ExplanationParameters
from openvino_xai.explanation.explanation_parameters import PostProcessParameters


# Compile model with XAI branch
compiled_model = ov.Core().compile_model(model_xai, "CPU")

# User's code that creates a callable model_inferrer with InferenceResult output
def model_inferrer(image: np.ndarray) -> InferenceResult:
    # Custom inference pipeline
    image_processed = preprocess(image)
    result = compiled_model([image_processed])
    logits = postprocess(result["logits"])
    raw_saliency_map = result["saliency_map"]  # "saliency_map" is an additional output added during insertion

    # Create InferenceResult object
    inference_result = InferenceResult(prediction=logits, saliency_map=raw_saliency_map)
    return inference_result

# Create explanation parameters, default parameter values are highlighted below
explanation_parameters = ExplanationParameters(
    explain_mode=ExplainMode.WHITEBOX,  # by default, run white-box XAI
    target_explain_group=TargetExplainGroup.PREDICTIONS,  # by default, explains only predicted classes
    post_processing_parameters=PostProcessParameters(overlay=True),  # by default, saliency map overlays over image
)
# Generate processed saliency map via .explain(model_inferrer, image) call
explanation = ovxai.explain(
    model_inferrer=model_inferrer,
    data=cv2.imread("path/to/image.jpg"),
    explanation_parameters=explanation_parameters,
)
explanation: ExplanationResult
explanation.saliency_map: Dict[int: np.ndarray]  # key - class id, value - processed saliency map e.g. 3x354x500
```

# Usage in black-box mode

## Explanation: generate explanation

```python
# Create original model
model: ov.Model
model = ov.Core().read_model("path/to/model.xml")

# Compile original model (no XAI branch inserted)
compiled_model = ov.Core().compile_model(model, "CPU")

# User's code that creates a callable model_inferrer
def model_inferrer(image: np.ndarray) -> InferenceResult:
    # Custom inference pipeline
    image_processed = preprocess(image)
    result = compiled_model([image_processed])
    logits = postprocess(result["logits"])

    # Create InferenceResult object, w/o saliency map.
    # Saliency map can be available in InferenceResult, but will be ignored when explain_mode=ExplainMode.BLACKBOX
    inference_result = InferenceResult(prediction=logits, saliency_map=None)
    return inference_result

# Generate explanation
explanation_parameters = ExplanationParameters(
    explain_mode=ExplainMode.BLACKBOX,  # Black-box XAI method will be used under .explain() call
)
explanation = ovxai.explain(
    model_inferrer=model_inferrer,
    data=cv2.imread("path/to/image.jpg"),
    explanation_parameters=explanation_parameters,
)
explanation: ExplanationResult
explanation.saliency_map: Dict[int: np.ndarray]  # key - class id, value - processed saliency map e.g. 354x500x3
```

See more usage scenarios in [examples](./examples).

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

# Scope of explained models

Models from [Pytorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) are used
for classification benchmark.

## White-box (fast, model-dependent)

### Classification

We benchmarked white-box explanation (using ReciproCAM explain method) using 528 models.
Currently, we support only CNN-based architectures in white-box mode,
transformers will be supported in the upcoming weeks.

For more details (statistic, model list, samples of generated saliency maps) see
[#11](https://github.com/intel-sandbox/openvino_xai/pull/11).

## Black-box (slow, model-agnostic)

### Classification

We benchmarked black-box explanation (using RISE explain method) using 528 CNN models and 115 transformer-based models.
Black-box explainer support all types of models that output logits (e.g. CNNs, transformers, etc.).

For more details (statistic, model list, samples of generated saliency maps) see
[#23](https://github.com/intel-sandbox/openvino_xai/pull/23).
