# OpenVINO-XAI: Classification explanation

This notebook shows an example how to use OpenVINO-XAI.

**OpenVINO-XAI** library is a tool that provides a suite of Explainable AI (XAI) algorithms for explanation of
[OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

It depicts a heatmap with areas of interest where neural network (classification or detection) focuses before making a decision. 

Example: Saliency map for `person` class for EfficientV2 classification model:

![Saliency Map Example](../../docs/images/saliency_map_person.png)

## Notebook Contents

The tutorial consists of the following steps:

- Update IR model with XAI branch to use it in own pipelines
- Create CustomInferrer to infer model and receive outputs 
- Explain model with White Box Explainer (updating original IR)
- Explain model with the Black Box explainer 

## Installation Instructions

```python
# Create virtual env
!python3 -m venv .ovxai

# Activate virtual env
!source .ovxai/bin/activate

# Optional for timm models usage
# Install torch (update CUDA version below if needed)
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Package installation
%pip install -q ..

```