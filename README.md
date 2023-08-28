# OpenVINO-XAI
Portable library for OpenVINO™ IR model explanation.

# Installation
- Editable mode (for development):
```bash
pip install -e .[tests]
```
- Package mode (for normal use):
```bash
pip install .
```

# Usage
### E2E model explanation
```python
# Patching IR model with XAI branch and using ModelAPI as inference framework
mapi_model_wrapper = XAIClassificationModel.create_model("path/to/model.xml")
explanation = WhiteBoxExplainer(mapi_model_wrapper).explain(cv2.imread("path/to/image.jpg"))
```
### Updating IR model
```python
# Embedding XAI into the model graph, no actual inference performed
# Afterwards, user suppose to use his/her own inference pipeline to get explanations along with the regular model outputs
ir_model_with_xai = XAIClassificationModel.insert_xai_into_native_ir("path/to/model.xml")
```
See more usage examples [here](./examples).
