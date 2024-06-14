# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Callable, List

import openvino.runtime as ov


class ModelType(Enum):
    """Enum representing the different model types."""

    CNN = "cnn"
    TRANSFORMER = "transformer"


class IRParser:
    """Parser parse OV IR model."""

    @staticmethod
    def get_logit_node(model: ov.Model, output_id: int = 0) -> ov.Node:
        logit_node = model.get_output_op(output_id).input(0).get_source_output().get_node()
        return logit_node

    @staticmethod
    def get_node_by_condition(ops: List[ov.Node], condition: Callable, k: int = 1):
        """Returns k-th node, which satisfies the condition."""
        for op in ops:
            if condition(op):
                k -= 1
                if k == 0:
                    return op

    @classmethod
    def _is_conv_node_w_spacial_size(cls, op: ov.Node) -> bool:
        if op.get_type_name() != "Convolution":
            return False
        if not cls._has_spacial_size(op):
            return False
        return True

    @classmethod
    def _is_concat_node_w_non_constant_inputs(cls, op):
        if op.get_type_name() != "Concat":
            return False
        input_nodes = op.inputs()
        for input_node in input_nodes:
            if input_node.get_source_output().get_node().get_type_name() == "Constant":
                return False
        return True

    @classmethod
    def _is_pooling_node_wo_spacial_size(cls, op: ov.Node) -> bool:
        if "Pool" not in op.get_friendly_name():
            return False
        if cls._has_spacial_size(op):
            return False
        return True

    @staticmethod
    def _is_op_w_single_spacial_output(op: ov.Node) -> bool:
        if op.get_type_name() == "Constant":
            return False
        if len(op.outputs()) > 1:
            return False
        node_out_shape = op.output(0).partial_shape
        if node_out_shape.rank.get_length() != 4:
            return False
        if not (node_out_shape[0].is_dynamic or node_out_shape[0].get_length() == 1):
            return False
        c = node_out_shape[1].get_length()
        h = node_out_shape[2].get_length()
        w = node_out_shape[3].get_length()
        if not (1 < h < c and 1 < w < c):
            return False
        return True

    @staticmethod
    def _has_spacial_size(node: ov.Node, output_id: int = 0) -> bool:
        node_out_shape = node.output(output_id).partial_shape

        # NCHW
        h = node_out_shape[2].get_length()
        w = node_out_shape[3].get_length()
        # NHWC
        h_ = node_out_shape[1].get_length()
        w_ = node_out_shape[2].get_length()
        return (h != 1 and w != 1) or (h_ != 1 and w_ != 1)

    @staticmethod
    def _is_add_node_w_two_non_constant_inputs(op: ov.Node):
        if op.get_type_name() != "Add":
            return False
        input_nodes = op.inputs()
        for input_node in input_nodes:
            if len(input_node.get_partial_shape()) != 3:
                return False
            if input_node.get_source_output().get_node().get_type_name() == "Constant":
                return False
            if input_node.get_source_output().get_node().get_type_name() == "Convert":
                return False
        return True


class IRParserCls(IRParser):
    """ParserCls parse classification OV IR model."""

    # TODO: use OV pattern matching functionality
    # TODO: separate for CNNs and ViT

    @classmethod
    def get_logit_node(cls, model: ov.Model, output_id=0, search_softmax=False) -> ov.Node:
        if search_softmax:
            reversed_ops = model.get_ordered_ops()[::-1]
            softmax_node = cls.get_node_by_condition(reversed_ops, lambda x: x.get_type_name() == "Softmax")
            if softmax_node and len(softmax_node.get_output_partial_shape(0)) == 2:
                logit_node = softmax_node.input(0).get_source_output().get_node()
                return logit_node

        logit_node = model.get_output_op(output_id).input(0).get_source_output().get_node()
        return logit_node

    @classmethod
    def get_target_node(
        cls,
        model: ov.Model,
        model_type: ModelType | None = None,
        target_node_name: str | None = None,
        k: int = 1,
    ) -> ov.Node:
        """
        Returns target node.
        Target node - node after which XAI branch will be inserted,
        i.e. output of the target node is used to generate input for the downstream XAI branch.
        """
        if target_node_name:
            reversed_ops = model.get_ordered_ops()[::-1]
            target_node = cls.get_node_by_condition(reversed_ops, lambda x: x.get_friendly_name() == target_node_name)
            if target_node is not None:
                return target_node
            raise ValueError(f"Cannot find {target_node_name} node.")

        if model_type == ModelType.CNN:
            # Make an attempt to search for last node with spacial dimensions
            reversed_ops = model.get_ordered_ops()[::-1]
            last_op_w_spacial_output = cls.get_node_by_condition(reversed_ops, cls._is_op_w_single_spacial_output, k)
            if last_op_w_spacial_output is not None:
                return last_op_w_spacial_output

            # Make an attempt to search for last backbone node via post_target_node
            post_target_node = cls.get_post_target_node(model)
            target_node = post_target_node.input(0).get_source_output().get_node()  # type: ignore
            if cls._has_spacial_size(target_node):
                return target_node

        if model_type == ModelType.TRANSFORMER:
            reversed_ops = model.get_ordered_ops()[::-1]
            target_node = cls.get_node_by_condition(reversed_ops, cls._is_add_node_w_two_non_constant_inputs, k)
            if target_node is not None:
                return target_node

        raise RuntimeError("Cannot find output backbone_node in auto mode, please provide target_layer.")

    @classmethod
    def get_post_target_node(
        cls,
        model,
        model_type: ModelType | None = None,
        target_node_name: str | None = None,
        target_node_output_id: int = 0,
    ) -> List[ov.Node]:
        if target_node_name:
            target_node = cls.get_target_node(model, model_type, target_node_name)
            target_node_outputs = target_node.output(target_node_output_id).get_target_inputs()
            post_target_nodes = [target_node_output.get_node() for target_node_output in target_node_outputs]
            return post_target_nodes

        if model_type == ModelType.CNN:
            # Make an attempt to search for a last pooling node
            reversed_ops = model.get_ordered_ops()[::-1]
            last_pooling_node = cls.get_node_by_condition(reversed_ops, cls._is_pooling_node_wo_spacial_size)
            if last_pooling_node is not None:
                return [last_pooling_node]

        raise RuntimeError("Cannot find first head node in auto mode, please explicitly provide input parameters.")

    @classmethod
    def get_first_conv_node(cls, model):
        ops = model.get_ordered_ops()
        first_conv_node = cls.get_node_by_condition(ops, cls._is_conv_node_w_spacial_size)
        if first_conv_node is not None:
            return first_conv_node

        raise RuntimeError("Cannot find first convolution node in auto mode.")

    @classmethod
    def get_first_concat_node(cls, model):
        ops = model.get_ordered_ops()
        first_concat_node = cls.get_node_by_condition(ops, cls._is_concat_node_w_non_constant_inputs)
        if first_concat_node is not None:
            return first_concat_node

        raise RuntimeError("Cannot find first concat node in auto mode.")
