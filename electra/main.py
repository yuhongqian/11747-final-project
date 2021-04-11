import os
import torch
import random
import logging
import numpy as np
import argparse
from data import MutualDataset, mutual_collate
from torch.utils.data import DataLoader
from runner import Trainer
from transformers import ElectraTokenizer, ElectraForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../MuTual/data/mutual")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of gpus to use for distributed training.")
    parser.add_argument("--output_dir", default="electra")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # TODO enable multi-gpu training if necessary
    train_dataset = MutualDataset(args.data_dir, "train", tokenizer) if args.train else None
    dev_dataset = MutualDataset(args.data_dir, "dev", tokenizer) if args.eval else None
    # TODO: add test_dataset if we want to submit to leaderboard

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8,
                                  collate_fn=mutual_collate(mode="train")) if train_dataset is not None else None
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=8,
                                collate_fn=mutual_collate(mode="dev")) if dev_dataset is not None else None

    if args.train:
        logging.info("Start training...")
        trainer = Trainer(args, model, device, train_dataloader, dev_dataloader)
        trainer.train()


if __name__ == '__main__':
    main()