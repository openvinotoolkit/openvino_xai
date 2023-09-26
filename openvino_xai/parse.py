# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import openvino


class IRParser:
    """Parser parse OV IR model."""

    @staticmethod
    def get_logit_node(model: openvino.runtime.Model, output_id: int = 0) -> openvino.runtime.Node:
        logit_node = (
            model.get_output_op(output_id)
            .input(0)
            .get_source_output()
            .get_node()
        )
        return logit_node


class IRParserCls(IRParser):
    """ParserCls parse classification OV IR model."""

    @staticmethod
    def get_logit_node(model: openvino.runtime.Model, output_id=0) -> openvino.runtime.Node:
        nodes = model.get_ops()
        softmax_node = None
        for op in nodes:
            if "Softmax" == op.get_type_name():
                softmax_node = op
        if softmax_node:
            logit_node = softmax_node.input(0).get_source_output().get_node()
            return logit_node

        logit_node = (
            model.get_output_op(output_id)
            .input(0)
            .get_source_output()
            .get_node()
        )
        return logit_node

    @staticmethod
    def get_output_backbone_node(
            model: openvino.runtime.Model, output_backbone_node_name: Optional[str] = None
    ) -> openvino.runtime.Node:
        if output_backbone_node_name:
            for op in model.get_ordered_ops():
                if op.get_friendly_name() == output_backbone_node_name:
                    return op
            raise ValueError(f"Cannot find {output_backbone_node_name} node.")

        # Make an attempt to use heuristics
        first_head_node = IRParserCls.get_input_head_node(model)
        output_backbone_node = first_head_node.input(0).get_source_output().get_node()
        return output_backbone_node

    @staticmethod
    def get_input_head_node(
            model, output_backbone_node_name: Optional[str] = None, output_backbone_id: int = 0
    ) -> openvino.runtime.Node:
        if output_backbone_node_name:
            output_backbone_node = IRParserCls.get_output_backbone_node(model, output_backbone_node_name)
            target_inputs = output_backbone_node.output(output_backbone_id).get_target_inputs()
            target_input_nodes = [target_input.get_node() for target_input in target_inputs]
            assert len(target_input_nodes) == 1, "Support only single target input."
            return target_input_nodes[0]

        # Apply heuristic - pick the last pooling layer
        # TODO: add more heuristics, e.g. check node type, not name
        for op in model.get_ordered_ops()[::-1]:
            if "Pool" in op.get_friendly_name():
                return op

        raise RuntimeError("Cannot find required target node in auto mode, please provide target_layer "
                           "in explain_parameters.")
