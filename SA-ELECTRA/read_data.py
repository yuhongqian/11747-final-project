import os
import torch
import random
import logging
import numpy as np
import argparse
from torch import nn  
from torch.utils.data import DataLoader
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, ElectraPreTrainedModel, ElectraModel
from data import MutualDataset
from transformers import ElectraTokenizer, ElectraModel
import torch
import torch
import os
import json
from torch.utils.data import Dataset
import pickle
import pdb


def add_eo_tokens(string):

        def find(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]
        z = string
        f = find(z, ':')
        new_str = ''
        for i in range(len(f)):
        
            if i != len(f) - 1:
                new_str = new_str + z[:find(z, ':')[i+1] - 2] + " " + '[EOU]' + " " + '[EOT] '
            else:
                new_str = new_str + z[f[i] - 2:] + " " + " " + '[EOU]' + " " + '[EOT] '

        return new_str


def get_speaker_ids(dialog, tokenizer, options): 

    z = dialog 
    inputs = tokenizer(text=add_eo_tokens(z), text_pair=options, padding="max_length",
                                      truncation="only_first", max_length=512, return_tensors="pt")

    # 30522, 30523
    
    speaker_ids = torch.zeros(inputs['input_ids'].shape, dtype=torch.long)
    turn = 0
    ind = 0
    reached_last_token = False
    for i in inputs['input_ids'][0,:]:

        if inputs['attention_mask'][0,ind].item() == 1:
            if i.item == 102: reached_last_token = True
            
            if reached_last_token == False:
                if i.item() == 30523: 
                    speaker_ids[0,ind] = turn
                    if turn == 0: turn = 1
                    else: turn = 0
                else:

                    speaker_ids[0,ind] = turn
            else:
                speaker_ids[0,-1] = speaker_ids[0,-2]
                last_speaker = speaker_ids[0,-1]
                speaker_ids[0, ind] = last_speaker
        
        else:
            speaker_ids[0, ind] = 0
        ind += 1

    inputs['speaker_ids'] = speaker_ids
    return inputs
    
    
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
