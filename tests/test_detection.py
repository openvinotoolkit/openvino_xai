from openvino_xai.insertion.insertion_parameters import DetectionInsertionParameters
from openvino_xai.common.parameters import XAIMethodType


def test_detection_explain_parameters():
    det_explain_params = DetectionInsertionParameters(["target_layer_name"], [5, 5, 5])
    assert det_explain_params.target_layer == ["target_layer_name"]
    assert det_explain_params.num_anchors == [5, 5, 5]
    assert det_explain_params.saliency_map_size == (13, 13)
    assert det_explain_params.embed_normalization
    assert det_explain_params.explain_method_type == XAIMethodType.DETCLASSPROBABILITYMAP
