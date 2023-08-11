# openvino_xai

```python
pip install -r requirements.txt
python setup.py develop
```

## Model retrieving

```python
# OTX classification model are downloaded and stored in .otx_models folder
pytest tests/test_classification.py
```

## Classification model explainer

```python
python examples/run_classification.py otx_models/mlc_efficient_b0_voc.xml images/cute-cat.jpg --output multilabel_saliency_map
```

## Quality checks

```python
pre-commit run -a
```
