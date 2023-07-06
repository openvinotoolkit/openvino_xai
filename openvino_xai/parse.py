class IRParser:
    @staticmethod
    def get_logit_node(model, output_id=0):
        logit_node = (
            model.get_output_op(output_id)
            .input(0)
            .get_source_output()
            .get_node()
        )
        return logit_node


class IRParserCls(IRParser):
    @staticmethod
    def get_output_backbone_node(model):
        # output_backbone_node_name = "/backbone/conv/conv.2/Div"  # mnet_v3
        # output_backbone_node_name = "/backbone/features/final_block/activate/Mul"  # effnet
        # for op in model.get_ordered_ops():
        #     if op.get_friendly_name() == output_backbone_node_name:
        #         return op

        first_head_node = IRParserCls.get_first_head_node(model)
        output_backbone_node = first_head_node.input(0).get_source_output().get_node()
        return output_backbone_node

    @staticmethod
    def get_first_head_node(model):
        # first_head_node_name = "/neck/gap/GlobalAveragePool"  # effnet and mnet_v3
        # for op in model.get_ordered_ops():
        #     if op.get_friendly_name() == first_head_node_name:
        #         return op

        for op in model.get_ordered_ops()[::-1]:
            if "GlobalAveragePool" in op.get_friendly_name():
                return op
