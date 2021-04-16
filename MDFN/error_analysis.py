from __future__ import absolute_import, division, print_function
import datetime
import argparse
import csv
import logging
import os
import random
import sys
import pickle
import numpy as np
import torch
import json
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import glob

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import (BertConfig, BertForMultipleChoice, BertTokenizer,
                            ElectraConfig, ElectraTokenizer, RobertaConfig, RobertaTokenizer, RobertaForMultipleChoice)
from modeling import (ElectraForMultipleChoicePlus, Baseline, BertBaseline, RobertaBaseline, BertForMultipleChoicePlus, RobertaForMultipleChoicePlus)
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
import re
import os
from run_MDFN import convert_examples_to_features, MuTualProcessor, select_field, compute_metrics

logger = logging.getLogger(__name__)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "output_mutual_electra_old/"
data_dir = 'datasets/mutual/'

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoicePlus, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoicePlus, RobertaTokenizer),
    'electra': (ElectraConfig, ElectraForMultipleChoicePlus, ElectraTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES['electra']

    
config = config_class.from_pretrained(model_path)
tokenizer = tokenizer_class.from_pretrained(model_path)
model = model_class.from_pretrained(model_path)
model.to(device)

mutual_data = MuTualProcessor()
filenames, eval_examples = mutual_data.get_dev_examples(data_dir)


label_list = mutual_data.get_labels()

eval_features = convert_examples_to_features(
                eval_examples, label_list, 128, 20, tokenizer, "classification")

logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", 1)
all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
all_sep_pos = torch.tensor(select_field(eval_features, 'sep_pos'), dtype=torch.long)
all_turn_ids = torch.tensor(select_field(eval_features, 'turn_ids'), dtype = torch.long)


all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)


eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sep_pos, all_turn_ids, all_label_ids)
# Run prediction for full data

eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

model.eval()
eval_loss = 0
nb_eval_steps = 0
preds = None

for batch in tqdm(eval_dataloader, desc="Evaluating"):

    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': None, # XLM don't use segment_ids
                'sep_pos': batch[3],
                'turn_ids': batch[4],
                'labels': batch[5]}

        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.detach().mean().item()
    
    nb_eval_steps += 1
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)


preds_class = np.argmax(preds, axis=1)

print(preds_class, out_label_ids)

error_analysis_dict = {}

for i in range(len(preds_class)):

    if preds_class[i] != out_label_ids[i]:
        error_analysis_dict[filenames[i]] = [preds_class[i], out_label_ids[i]]

import pickle
f = open("error_analysis_dict.pkl","wb")
pickle.dump(error_analysis_dict,f)
f.close()