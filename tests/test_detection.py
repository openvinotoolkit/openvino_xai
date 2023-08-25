from openvino_xai.parameters import DetectionExplainParametersWB


def test_detection_explain_parameters():
    det_explain_params = DetectionExplainParametersWB(["target_layer_name"], [5, 5, 5])
    assert det_explain_params.target_layer == ["target_layer_name"]
    assert det_explain_params.num_anchors == [5, 5, 5]
    assert det_explain_params.saliency_map_size == (13, 13)
    assert det_explain_params.embed_normalization
    assert det_explain_params.explain_method_name == "detclassprobabilitymap"
