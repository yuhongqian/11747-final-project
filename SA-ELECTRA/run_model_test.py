import os
import torch
import random
import logging
import numpy as np
import argparse
from read_data import MutualDataset, mutual_collate
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, ElectraModel
from speaker_aware_embeddings import SpeakerAwareElectraModelForSequenceClassification
from testing import Tester


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="mutual/")
    parser.add_argument("--checkpoint")
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
    parser.add_argument("--constrasitve", action="store_true") 
    parser.add_argument("--speaker_embeddings", action="store_true")
    
    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

config = ElectraConfig.from_pretrained(args.model_name, num_labels=1)   # 1 label for regression
tokenizer = ElectraTokenizer.from_pretrained(args.model_name, do_lower_case=True) 

model = SpeakerAwareElectraModelForSequenceClassification(args.model_name, config, 2)

model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu'))['model'])
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


dev_dataset = MutualDataset('mutual/', "dev", tokenizer)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=mutual_collate) 

tester = Tester(args, model, device, dev_dataset, dev_dataloader)
tester.test()