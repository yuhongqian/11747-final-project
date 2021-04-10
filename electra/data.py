import torch
import os
import json
from torch.utils.data import Dataset
import pickle
import pdb

ANSWER_IDX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}


def load_data(mode_dir, mode, tokenizer):
    tokenized_file = os.path.join(mode_dir, f"{mode}.tokenized.pkl")
    if not os.path.exists(tokenized_file):
        json_file = os.path.join(mode_dir, f"{mode}.json")
        if not os.path.exists(json_file):
            raw_data = convert_dir_to_json(mode_dir, json_file)
        else:
            with open(json_file, "r") as f:
                raw_data = json.load(f)
        return get_tokenized_data(raw_data, tokenizer, tokenized_file)
    else:
        return json.load(tokenized_file)


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


def get_tokenized_data(raw_data, tokenizer, tokenized_file):
    tokenized_data = []
    for data_id, raw_example in raw_data.items():
        answer = ANSWER_IDX[raw_example["answers"]]
        for idx, option in enumerate(raw_example["options"]):
            curr_example = dict()
            curr_example["id"] = raw_example["id"]
            curr_example["answer"] = answer
            curr_example["option_id"] = idx
            curr_example["label"] = 1.0 if idx == answer else 0.0   # float for MSE loss
            # dialog history is put first, following electra-dapo paper
            tokenized = tokenizer(text=raw_example["article"], text_pair=option,
                                                  truncation="only_first", max_length=512, return_tensors="pt")
            curr_example["input_ids"] = tokenized["input_ids"]
            curr_example["token_type_ids"] = tokenized["token_type_ids"]
            curr_example["attention_mask"] = tokenized["attention_mask"]
            tokenized_data.append(curr_example)
    with open(tokenized_file, "wb") as f:
        pickle.dump(tokenized_data, f)
    return tokenized_data


class MutualDataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer):
        if mode not in {"dev", "test", "train"}:
            raise ValueError("Incorrect dataset mode.")

        mode_dir = os.path.join(data_dir, mode)
        self.data = load_data(mode_dir, mode, tokenizer)

    def __getitem__(self, item):
        # TODO tokenize data here
        return self.data[item]

    def __len__(self):
        return len(self.data)