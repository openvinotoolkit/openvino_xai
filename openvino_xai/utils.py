# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

logger = logging.getLogger("openvino_xai")
logger.setLevel(logging.INFO)


def reorder_sal_map(saliency_map, hierarchical_info, labels):
    """Reorder saliency maps from hierarchical model output to be adjusted to label-schema labels order."""
    hierarchical_idx = []
    for head_idx in range(hierarchical_info["num_multiclass_heads"]):
        logits_begin, logits_end = hierarchical_info["head_idx_to_logits_range"][str(head_idx)]
        for logit in range(0, logits_end - logits_begin):
            label_str = hierarchical_info["all_groups"][head_idx][logit]
            label_idx = hierarchical_info["label_to_idx"][label_str]
            hierarchical_idx.append(label_idx)

    if hierarchical_info["num_multilabel_classes"]:
        logits_begin = hierarchical_info["num_single_label_classes"]
        logits_end = len(labels)
        for logit_idx, logit in enumerate(range(0, logits_end - logits_begin)):
            label_str_idx = hierarchical_info["num_multiclass_heads"] + logit_idx
            label_str = hierarchical_info["all_groups"][label_str_idx][0]
            label_idx = hierarchical_info["label_to_idx"][label_str]
            hierarchical_idx.append(label_idx)

    reordered_map = {}
    for i, h_idx in enumerate(hierarchical_idx):
        reordered_map[h_idx] = saliency_map[i]
    return dict(sorted(reordered_map.items()))
