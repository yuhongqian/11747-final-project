import os
import csv
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import pdb

IDX_TO_ANSWER = ["A", "B", "C", "D"]

class Tester:
    def __init__(self, args, model, device, dataset, dataloader):
        self.args = args
        self.model = model
        self.device = device
        self.dataset = dataset
        self.dataloader = dataloader

    def test(self):
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for example in tqdm(self.dataloader, desc="Test"):
                #print("yo!")
                input_ids, token_type_ids, attention_mask, speaker_ids, labels = example
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                speaker_ids    = speaker_ids.to(self.device)
                logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, speaker_ids=speaker_ids)
                scores = F.sigmoid(logits)
                del input_ids, token_type_ids, attention_mask, speaker_ids
                torch.cuda.empty_cache()
                all_scores.extend(list(scores.detach().cpu().numpy()))     # TODO: change to np?
        # pdb.set_trace()
        prev_id = None
        curr_scores = []    # (score, option_id)
        with open(os.path.join(self.args.output_dir, "output.txt"), "w") as f1, \
                open(os.path.join(self.args.output_dir, "output_w_scores.txt"), "w") as f2:
            writer1 = csv.writer(f1, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer2 = csv.writer(f2, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for i, example in enumerate(self.dataset):
                curr_id = example["id"]
                if prev_id is not None and curr_id != prev_id:
                    curr_scores.sort(reverse=True)
                    row = [prev_id]
                    answers = [IDX_TO_ANSWER[option_id] for (_, option_id) in curr_scores]
                    answers_w_scores = [(IDX_TO_ANSWER[option_id], score) for (score, option_id) in curr_scores]
                    writer1.writerow(row + answers)
                    writer2.writerow(row + answers_w_scores)
                    curr_scores = []
                curr_scores.append((all_scores[i][0], example["option_id"]))
                prev_id = curr_id
            curr_scores.sort(reverse=True)
            row = [prev_id]
            answers = [IDX_TO_ANSWER[option_id] for (_, option_id) in curr_scores]
            answers_w_scores = [(IDX_TO_ANSWER[option_id], score) for (score, option_id) in curr_scores]
            writer1.writerow(row + answers)
            writer2.writerow(row + answers_w_scores)