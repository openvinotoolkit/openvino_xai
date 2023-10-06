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

    @staticmethod
    def get_node_by_condition(ops, condition):
        for op in ops:
            if condition(op):
                return op

    @classmethod
    def _is_conv_node_w_spacial_size(cls, op: openvino.runtime.Node) -> bool:
        if op.get_type_name() != "Convolution":
            return False
        if not cls._has_spacial_size(op):
            return False
        return True

    @classmethod
    def _is_pooling_node_wo_spacial_size(cls, op: openvino.runtime.Node) -> bool:
        if "Pool" not in op.get_friendly_name():
            return False
        if cls._has_spacial_size(op):
            return False
        return True

    @staticmethod
    def _is_op_w_single_spacial_output(op: openvino.runtime.Node) -> bool:
        if op.get_type_name() == "Constant":
            return False
        if len(op.outputs()) > 1:
            return False
        node_output_shape = op.output(0).partial_shape
        if node_output_shape.rank.get_length() != 4:
            return False
        c, h, w, = node_output_shape[1].get_length(), node_output_shape[2].get_length(), node_output_shape[3].get_length()
        return 1 < h < c and 1 < w < c

    @staticmethod
    def _has_spacial_size(node: openvino.runtime.Node, output_id: int = 0) -> Optional[bool]:
        node_output_shape = node.output(output_id).partial_shape

        # NCHW
        h, w, = node_output_shape[2].get_length(), node_output_shape[3].get_length()
        # NHWC
        h_, w_, = node_output_shape[1].get_length(), node_output_shape[2].get_length()
        return (h != 1 and w != 1) or (h_ != 1 and w_ != 1)


class IRParserCls(IRParser):
    """ParserCls parse classification OV IR model."""

    @classmethod
    def get_logit_node(cls, model: openvino.runtime.Model, output_id=0) -> openvino.runtime.Node:
        reversed_ops = model.get_ordered_ops()[::-1]
        softmax_node = cls.get_node_by_condition(reversed_ops, lambda x: x.get_type_name() == "Softmax")
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
            reversed_ops = model.get_ordered_ops()[::-1]
            output_backbone_node = cls.get_node_by_condition(
                reversed_ops, lambda x: x.get_friendly_name() == output_backbone_node_name
            )
            if output_backbone_node is not None:
                return output_backbone_node
            raise ValueError(f"Cannot find {output_backbone_node_name} node.")

        # Make an attempt to search for last node with spacial dimensions
        reversed_ops = model.get_ordered_ops()[::-1]
        last_op_w_spacial_output = cls.get_node_by_condition(reversed_ops, cls._is_op_w_single_spacial_output)
        if last_op_w_spacial_output is not None:
            return last_op_w_spacial_output

        # Make an attempt to search for last backbone node via a first head node
        first_head_node = cls.get_input_head_node(model)
        output_backbone_node = first_head_node.input(0).get_source_output().get_node()
        if cls._has_spacial_size(output_backbone_node):
            return output_backbone_node

        raise RuntimeError(f"Cannot find output backbone_node in auto mode, please provide target_layer.")

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

        # Make an attempt to search for a last pooling node
        reversed_ops = model.get_ordered_ops()[::-1]
        last_pooling_node = cls.get_node_by_condition(reversed_ops, cls._is_pooling_node_wo_spacial_size)
        if last_pooling_node is not None:
            return last_pooling_node

        raise RuntimeError("Cannot find first head node in auto mode, please explicitly provide input parameters.")
