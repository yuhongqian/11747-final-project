import os
import torch
import random
import logging
import numpy as np
import argparse
from data import MutualDataset, mutual_collate, DapoDataset, dapo_collate
from torch.utils.data import DataLoader
from runner import Trainer, Tester
from transformers import BertConfig, BertForSequenceClassification, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../MuTual/data/mutual")
    parser.add_argument("--model_name")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of gpus to use for distributed training.")
    parser.add_argument("--output_dir", default="ckpts")
    parser.add_argument("--local_model_path", default=None, type=str)
    parser.add_argument("--numnet_model", default=None, type=str)
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

    config = ElectraConfig.from_pretrained(args.model_name, num_labels=1)   # 1 label for regression
    model = ElectraForSequenceClassification.from_pretrained(args.model_name, config=config)

    if args.numnet_model is not None:
        config = BertConfig.from_pretrained(args.model_name, num_labels=1)  # 1 label for regression
        model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)
        state_dicts = torch.load(args.numnet_model)
        if "model" in state_dicts:
            logging.info("Loading in mutual electra format state_dicts.")
            model.load_state_dict(state_dicts["model"], strict=False)
        else:
            logging.info("Loading model weights only.")
            model.load_state_dict(state_dicts)
        model.load_state_dict(torch.load(args.numnet_model), strict=False)
    elif args.local_model_path is not None:
        state_dicts = torch.load(args.local_model_path)
        model.load_state_dict(state_dicts["model"])

    tokenizer = ElectraTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # TODO enable multi-gpu training if necessary
    pretrain_train_dataset = DapoDataset(args.data_dir, "train", tokenizer) if args.pretrain else None
    pretrain_dev_dataset = DapoDataset(args.data_dir, "dev", tokenizer) if args.pretrain else None
    train_dataset = MutualDataset(args.data_dir, "train", tokenizer) if args.train else None
    dev_dataset = MutualDataset(args.data_dir, "dev", tokenizer) if args.eval or args.test else None
    # TODO: add test_dataset if we want to submit to leaderboard

    pretrain_train_dataloader = DataLoader(pretrain_train_dataset, batch_size=args.train_batch_size,
                                           shuffle=True, num_workers=8,
                                           collate_fn=dapo_collate) if pretrain_train_dataset is not None else None
    pretrain_dev_dataloader = DataLoader(pretrain_dev_dataset, batch_size=args.train_batch_size,
                                           shuffle=False, num_workers=8,
                                           collate_fn=dapo_collate) if pretrain_dev_dataset is not None else None
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8,
                                  collate_fn=mutual_collate) if train_dataset is not None else None
    # currently eval_batch_size = train_batch_size
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=8,
                                collate_fn=mutual_collate) if dev_dataset is not None else None

    if args.pretrain:
        logging.info("Start pretraining...")
        args.eval = True
        trainer = Trainer(args, model, device, pretrain_train_dataloader, pretrain_dev_dataloader)
        trainer.train()
        return  # fine-tuning should be done separately

    if args.train:
        logging.info("Start training...")
        trainer = Trainer(args, model, device, train_dataloader, dev_dataloader)
        trainer.train()

    # TODO: currently testing is on the dev set
    if args.test:
        logging.info("Start testing...")
        tester = Tester(args, model, device, dev_dataset, dev_dataloader)
        tester.test()



if __name__ == '__main__':
    main()