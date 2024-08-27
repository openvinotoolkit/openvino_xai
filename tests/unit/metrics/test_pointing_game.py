import logging

import numpy as np
import pytest

from openvino_xai.explainer.explanation import Explanation
from openvino_xai.metrics.pointing_game import PointingGame


class TestPointingGame:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.pointing_game = PointingGame()

    def test_pointing_game(self):
        saliency_map = np.zeros((3, 3), dtype=np.float32)
        saliency_map[1, 1] = 1

        ground_truth_bbox = [(1, 1, 1, 1)]
        score_result = self.pointing_game(saliency_map, ground_truth_bbox)
        assert score_result["pointing_game"] == 1

        ground_truth_bbox = [(0, 0, 0, 0)]
        score_result = self.pointing_game(saliency_map, ground_truth_bbox)
        assert score_result["pointing_game"] == 0

    def test_pointing_game_evaluate(self, caplog):
        pointing_game = PointingGame()
        explanation = Explanation(
            label_names=["cat", "dog"],
            targets=[0, 1],
            saliency_map={0: [[0, 1], [2, 3]], 1: [[0, 0], [0, 1]]},
        )
        explanations = [explanation]

        gt_bboxes = [{"cat": [(0, 0, 2, 2)], "dog": [(0, 0, 1, 1)]}]
        score_result = pointing_game.evaluate(explanations, gt_bboxes)
        assert score_result["pointing_game"] == 1.0

        # No hit for dog class saliency map, hit for cat class saliency map
        gt_bboxes = [{"cat": [(0, 0, 2, 2), (0, 0, 1, 1)], "dog": [(0, 0, 0, 0)]}]
        score_result = pointing_game.evaluate(explanations, gt_bboxes)
        assert score_result["pointing_game"] == 0.5

        # No ground truth bboxes for available saliency map classes
        gt_bboxes = [{"not-cat": [(0, 0, 2, 2)], "not-dog": [(0, 0, 0, 0)]}]
        with caplog.at_level(logging.INFO):
            score_result = pointing_game.evaluate(explanations, gt_bboxes)
        assert "Skip pointing game evaluation for this saliency map." in caplog.text
        assert score_result["pointing_game"] == 0.0

        # Ground truth bboxes / saliency maps number mismatch
        gt_bboxes = []
        with pytest.raises(AssertionError):
            score_result = pointing_game.evaluate(explanations, gt_bboxes)

        # No label names
        explanation = Explanation(
            label_names=None,
            targets=[0, 1],
            saliency_map={0: [[0, 1], [2, 3]], 1: [[0, 0], [0, 1]]},
        )
        explanations = [explanation]
        gt_bboxes = [{"cat": [(0, 0, 2, 2)], "dog": [(0, 0, 1, 1)]}]
        with pytest.raises(AssertionError):
            score_result = pointing_game.evaluate(explanations, gt_bboxes)
