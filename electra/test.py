import argparse
from data import MutualDataset
from transformers import ElectraTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../MuTual/data/mutual")
    return parser.parse_args()


def test_mutual_dataset(args):
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator", do_lower_case=True, use_fast=True)
    data = MutualDataset(args.data_dir, "train", tokenizer)
    assert(len(data) == 7088 * 4)


def main():
    args = parse_args()
    test_mutual_dataset(args)


if __name__ == '__main__':
    main()