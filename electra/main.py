import torch
import argparse
from data import MutualDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../MuTual/data/mutual")
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_dataset = MutualDataset(args, "train") if args.train else None
    dev_dataset = MutualDataset(args, "dev") if args.eval else None
    test_dataset = MutualDataset(args, "test") if args.test else None

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
