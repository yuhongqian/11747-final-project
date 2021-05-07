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


class ContrastiveDataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer):
        if mode not in {"train", "dev", "test"}:
            raise ValueError("Incorrect dataset mode.")

        self.mode = mode
        self.mode_dir = os.path.join(data_dir, mode)
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self):
        mode_dir, mode, tokenizer = self.mode_dir, self.mode, self.tokenizer
        tokenized_file = os.path.join(mode_dir, f"{mode}.contrast.tokenized.pkl")
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
            positive = raw_example["options"][answer]
            for idx, option in raw_example["options"]:
                if idx == answer:
                    continue
                curr_example = dict()
                curr_example["id"] = raw_example["id"]
                # positive example
                curr_example["positive"] = positive
                pos_tokenized = tokenizer(text=raw_example["article"], text_pair=positive, padding="max_length",
                                          truncation="only_first", max_length=512, return_tensors="pt")
                curr_example["pos_input_ids"] = pos_tokenized["input_ids"]
                curr_example["pos_token_type_ids"] = pos_tokenized["token_type_ids"]
                curr_example["pos_attention_mask"] = pos_tokenized["attention_mask"]
                # negative example
                curr_example["negative"] = option
                neg_tokenized = tokenizer(text=raw_example["article"], text_pair=option, padding="max_length",
                                   truncation="only_first", max_length=512, return_tensors="pt")
                curr_example["neg_input_ids"] = neg_tokenized["input_ids"]
                curr_example["neg_token_type_ids"] = neg_tokenized["token_type_ids"]
                curr_example["neg_attention_mask"] = neg_tokenized["attention_mask"]
                if idx == answer:
                    continue
                tokenized_data.append(curr_example)
        with open(tokenized_file, "wb") as f:
            pickle.dump(tokenized_data, f)
        return tokenized_data


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
                tokenized = tokenizer(text=raw_example["article"], text_pair=option, padding="max_length",
                                      truncation="only_first", max_length=512, return_tensors="pt")
                curr_example["input_ids"] = tokenized["input_ids"]
                curr_example["token_type_ids"] = tokenized["token_type_ids"]
                curr_example["attention_mask"] = tokenized["attention_mask"]
                tokenized_data.append(curr_example)
        with open(tokenized_file, "wb") as f:
            pickle.dump(tokenized_data, f)
        return tokenized_data


def mutual_collate(batch):
    input_ids = torch.stack([example["input_ids"] for example in batch]).squeeze()
    token_type_ids = torch.stack([example["token_type_ids"] for example in batch]).squeeze()
    attention_mask = torch.stack([example["attention_mask"] for example in batch]).squeeze()
    labels = torch.stack([example["label"] for example in batch]).unsqueeze(dim=1) \
        if batch[0]["label"] is not None else None
    return input_ids, token_type_ids, attention_mask, labels


def mutual_contrast_collate(batch):
    pos_input_ids = torch.stack([example["pos_input_ids"] for example in batch]).squeeze()
    pos_token_type_ids = torch.stack([example["pos_token_type_ids"] for example in batch]).squeeze()
    pos_attention_mask = torch.stack([example["pos_attention_mask"] for example in batch]).squeeze()
    neg_input_ids = torch.stack([example["neg_input_ids"] for example in batch]).squeeze()
    neg_token_type_ids = torch.stack([example["neg_token_type_ids"] for example in batch]).squeeze()
    neg_attention_mask = torch.stack([example["neg_attention_mask"] for example in batch]).squeeze()
    return pos_input_ids, pos_token_type_ids, pos_attention_mask, neg_input_ids, neg_token_type_ids, neg_attention_mask


class DapoDataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer):
        if mode not in {"train", "dev"}:
            raise ValueError("Incorrect dataset mode.")
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data_dir, mode, tokenizer = self.data_dir, self.mode, self.tokenizer
        tokenized_file = os.path.join(data_dir, f"{mode}.tokenized.pkl")
        if not os.path.exists(tokenized_file):
            data = []
            with open(os.path.join(data_dir, f"pretrain_{mode}.txt"), "r") as f:
                for l in f:
                    curr_example = json.loads(l)
                    tokenized = tokenizer(text=curr_example["conversation"], padding="max_length",
                                          truncation=True, max_length=512, return_tensors="pt")
                    curr_example["input_ids"] = tokenized["input_ids"]
                    curr_example["token_type_ids"] = tokenized["token_type_ids"]
                    curr_example["attention_mask"] = tokenized["attention_mask"]
                    curr_example["label"] = torch.tensor(curr_example["label"], dtype=torch.float32)
                    data.append(curr_example)
            with open(tokenized_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(tokenized_file, "rb") as f:
                data = pickle.load(f)
        return data


def dapo_collate(batch):
    return mutual_collate(batch)

