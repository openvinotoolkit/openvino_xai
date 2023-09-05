# OpenVINO-XAI
OpenVINO-XAI provides a suite of Explainable AI (XAI) algorithms for explanation of 
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

# Usage
## CLI usage

- Model retreival
```python
# OTX classification model are downloaded and stored in .otx_models folder
pytest tests/test_classification.py
```
- Classification model explainer

```python
python examples/run_classification.py otx_models/mlc_efficient_b0_voc.xml images/cute-cat.jpg --output multilabel_saliency_map
```

## E2E model explanation
```python
from openvino_xai.model import XAIClassificationModel
from openvino_xai.explain import WhiteBoxExplainer

# Create an OpenVINO™ ModelAPI model wrapper with XAI head inserted into the model graph
mapi_model_wrapper = XAIClassificationModel.create_model("path/to/model.xml")

# ModelAPI is used as an inference framework
# Explanation is generated during inference along with the regular model output
explanation = WhiteBoxExplainer(mapi_model_wrapper).explain(cv2.imread("path/to/image.jpg"))
```
### Updating IR model
```python
# Embedding XAI branch into the model graph, no actual inference performed
ir_model_with_xai = XAIClassificationModel.insert_xai_into_native_ir("path/to/model.xml")
# Now, user suppose to use his/her own inference pipeline to infer ir_model_with_xai
```
See more usage scenarios in [examples](./examples).


# Scope of explained models

## White-box explanation (fast, model-dependent)
### Classification
We support 542 models from [Pytorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models).
Currently, we support only CNN-based architectures in white-box mode (using ReciproCAM explain method), 
transformers will be supported in the upcoming weeks.

For more details (numbers, model list, samples of generated saliency maps) see 
[#11](https://github.com/intel-sandbox/openvino_xai/pull/11).

## Black-box (slow, model-agnostic)
### Classification
TODO: gz
