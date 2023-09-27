# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

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

    @classmethod
    def get_output_backbone_node(
            cls, model: openvino.runtime.Model, output_backbone_node_name: Optional[str] = None
    ) -> openvino.runtime.Node:
        if output_backbone_node_name:
            for op in model.get_ordered_ops():
                if op.get_friendly_name() == output_backbone_node_name:
                    return op
            raise ValueError(f"Cannot find {output_backbone_node_name} node.")

        # Make an attempt to search for a first_head_node
        first_head_node = cls.get_input_head_node(model)
        output_backbone_node = first_head_node.input(0).get_source_output().get_node()
        spacial_shape = cls._has_spacial_shape(output_backbone_node)
        if spacial_shape is not None:
            if not spacial_shape:
                # output_backbone_node suppose to have spacial dimensions
                raise RuntimeError(f"Cannot find output backbone_node in auto mode, please provide target_layer.")

        return output_backbone_node

    @classmethod
    def get_input_head_node(
            cls, model, output_backbone_node_name: Optional[str] = None, output_backbone_id: int = 0
    ) -> openvino.runtime.Node:
        if output_backbone_node_name:
            output_backbone_node = cls.get_output_backbone_node(model, output_backbone_node_name)
            target_inputs = output_backbone_node.output(output_backbone_id).get_target_inputs()
            target_input_nodes = [target_input.get_node() for target_input in target_inputs]
            assert len(target_input_nodes) == 1, "Support only single target input."
            return target_input_nodes[0]

        for op in model.get_ordered_ops()[::-1]:
            if cls._is_first_head_node(op):
                return op

        raise RuntimeError("Cannot find first head node in auto mode, please explicitly provide input parameters.")

    @classmethod
    def _is_first_head_node(cls, op: openvino.runtime.Node) -> bool:
        checks = {
            "last_pooling_layer": "Pool" in op.get_friendly_name() and not cls._has_spacial_shape(op),
        }
        return all(checks.values())

    @staticmethod
    def _has_spacial_shape(
            node: openvino.runtime.Node, output_id: int = 0, axis: Tuple[int] = (2, 3)
    ) -> Optional[bool]:
        node_output_shape = node.output(output_id).partial_shape
        h, w, = node_output_shape[axis[0]], node_output_shape[axis[1]]
        dynamic = h.is_dynamic or w.is_dynamic
        if not dynamic:
            return h.get_length() > 1 and w.get_length() > 1
