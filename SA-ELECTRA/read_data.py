import os
import torch
import random
import logging
import numpy as np
import argparse
from torch import nn  
from torch.utils.data import DataLoader
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, ElectraPreTrainedModel, ElectraModel
from transformers import ElectraTokenizer, ElectraModel
import torch
import torch
import os
import json
from torch.utils.data import Dataset
import pickle
import pdb

def get_speaker_ids(dialog, tokenizer, options): 

    z = dialog 
   
    inputs = tokenizer(text=z, text_pair=options, padding="max_length",
                                     truncation="only_first", max_length=512, return_tensors="pt")
    # 30522, 30523
    
    speaker_ids = torch.zeros(inputs['input_ids'].shape, dtype=torch.long)
    turn = 0
    ind = 0
    reached_last_token = False
    for i in inputs['input_ids'][0,:]:

        if inputs['attention_mask'][0,ind].item() == 1:
            if i.item() == 1024 and inputs['input_ids'][0,ind - 2] != 101: 
                if turn == 0: 
                    turn = 1
                else: 
                    turn = 0
                
                speaker_ids[0,ind] = turn
                speaker_ids[0,ind-1] = turn

            if i.item() == 102 and reached_last_token == False:
                if turn == 0: 
                    turn = 1
                else: 
                    turn = 0
                speaker_ids[0,ind] = turn

                reached_last_token = True
            
            else:
                speaker_ids[0,ind] = turn
        
            ind += 1

    inputs['speaker_ids'] = speaker_ids
    return inputs


config = ElectraConfig.from_pretrained("google/electra-small-discriminator", num_labels=1)   # 1 label for regression
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator", do_lower_case=True) 

ANSWER_IDX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

def convert_dir_to_json(mode_dir, json_path):
    raw_data = dict()   # "id" -> raw_data
    with open(json_path, "w") as fout:
        for fname in os.listdir(mode_dir):
            if not fname.endswith("txt"):
                continue
            with open(os.path.join(mode_dir, fname), "r") as fin:
                l = fin.readline()
                example = json.loads(l)
                raw_data[example["id"]] = example
        json.dump(raw_data, fout)
    return raw_data


class MutualDataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer):
        if mode not in {"train", "dev", "test"}:
            raise ValueError("Incorrect dataset mode.")

        self.mode = mode
        self.mode_dir = os.path.join(data_dir, mode)
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        mode_dir, mode, tokenizer = self.mode_dir, self.mode, self.tokenizer
        tokenized_file = os.path.join(mode_dir, f"{mode}.tokenized.pkl")
        if not os.path.exists(tokenized_file):
            json_file = os.path.join(mode_dir, f"{mode}.json")
            if not os.path.exists(json_file):
                raw_data = convert_dir_to_json(mode_dir, json_file)
            else:
                with open(json_file, "r") as f:
                    raw_data = json.load(f)
            return self.get_tokenized_data(raw_data, tokenized_file)
        else:
            with open(tokenized_file, "rb") as f:
                return pickle.load(f)

    def get_tokenized_data(self, raw_data, tokenized_file):
        tokenizer, mode = self.tokenizer, self.mode
        tokenized_data = []
        for data_id, raw_example in raw_data.items():
            answer = ANSWER_IDX[raw_example["answers"]]
            for idx, option in enumerate(raw_example["options"]):
                curr_example = dict()
                curr_example["id"] = raw_example["id"]
                curr_example["answer"] = answer
                curr_example["option_id"] = idx
                if mode == "test":
                    curr_example["label"] = None
                else:
                    curr_example["label"] = torch.tensor(1.0) if idx == answer else torch.tensor(
                        0.0)  # float for MSE loss
            
                # dialog history is put first, following electra-dapo paper


                tokenized = get_speaker_ids(raw_example["article"], tokenizer, option)

                curr_example["input_ids"] = tokenized["input_ids"]
                curr_example["token_type_ids"] = tokenized["token_type_ids"]
                curr_example["attention_mask"] = tokenized["attention_mask"]
                curr_example['speaker_ids'] = tokenized["speaker_ids"]
               
                tokenized_data.append(curr_example)
        with open(tokenized_file, "wb") as f:
            pickle.dump(tokenized_data, f)
        return tokenized_data

def mutual_collate(batch):
    input_ids = torch.stack([example["input_ids"] for example in batch]).squeeze()
    token_type_ids = torch.stack([example["token_type_ids"] for example in batch]).squeeze()
    attention_mask = torch.stack([example["attention_mask"] for example in batch]).squeeze()
    speaker_ids    = torch.stack([example['speaker_ids'] for example in batch]).squeeze()
    labels = torch.stack([example["label"] for example in batch]).unsqueeze(dim=1) \
        if batch[0]["label"] is not None else None
    return input_ids, token_type_ids, attention_mask, speaker_ids, labels


dat1 = MutualDataset('mutual/', 'train', tokenizer)
dat2 = MutualDataset('mutual/', 'dev', tokenizer)
