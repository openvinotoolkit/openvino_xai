import sys

import argparse

from openvino_xai.classification import InsertXAICls


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--save_path')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    ir_xai = InsertXAICls(args.model_path)  # , explain_algorithm="activationmap"
    print("\nOriginal model:\n", ir_xai.model_ori)
    ir_xai.generate_model_with_xai()
    print("\nModel with XAI inserted:\n", ir_xai.model_with_xai)
    if args.save_path:
        ir_xai.serialize_model_with_xai(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
