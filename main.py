import sys

import argparse

from openvino_xai.inserter import InsertXAICls, InsertXAIDet


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--save_path')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    # # # Classification example
    # ir_with_xai_generator = InsertXAICls(args.model_path)  # , explain_algorithm="activationmap"
    # print("\nOriginal model:\n", ir_with_xai_generator.model_ori)
    # model_with_xai = ir_with_xai_generator.generate_model_with_xai()
    # print("\nModel with XAI inserted:\n", model_with_xai)
    # if args.save_path:
    #     ir_with_xai_generator.serialize_model_with_xai(args.save_path)

    # Detection example
    ir_with_xai_generator = InsertXAIDet(args.model_path)
    print("\nOriginal model:\n", ir_with_xai_generator.model_ori)
    model_with_xai = ir_with_xai_generator.generate_model_with_xai()
    print("\nModel with XAI inserted:\n", model_with_xai)
    if args.save_path:
        ir_with_xai_generator.serialize_model_with_xai(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
