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
from speaker_aware_embeddings import SpeakerAwareElectraEmbeddings
from transformers import BertConfig, BertForSequenceClassification, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, ElectraModel


IDX_TO_ANSWER = ["A", "B", "C", "D"]


class Trainer:
    def __init__(self, args, model, device, train_dataloader, dev_dataloader):
        self.args = args
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.criterion    = nn.MSELoss(reduction="mean")

    def train(self):
        self.model.train()
        writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir, "runs"))
        optimizer = Adam(params=self.model.parameters(), lr=self.args.lr, eps=1e-8, weight_decay=self.args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=75000)
        global_step = 0
        best_eval_loss = None
        for epoch in tqdm(range(self.args.epochs), desc="Epoch"):
            train_loss = 0
            for step, example in tqdm(enumerate(self.train_dataloader), desc="Train"):

                if self.args.constrasitve:
                    pos_input_ids, pos_token_type_ids, pos_attention_mask, neg_input_ids, neg_token_type_ids, \
                    neg_attention_mask = example
                    pos_input_ids = pos_input_ids.to(self.device)
                    pos_token_type_ids = pos_token_type_ids.to(self.device)
                    pos_attention_mask = pos_attention_mask.to(self.device)
                    neg_input_ids = neg_input_ids.to(self.device)
                    neg_token_type_ids = neg_token_type_ids.to(self.device)
                    neg_attention_mask = neg_attention_mask.to(self.device)
                    loss = self.model(pos_input_ids, pos_token_type_ids, pos_attention_mask,
                                      neg_input_ids, neg_token_type_ids, neg_attention_mask)
                
                if self.args.speaker_embeddings:
                    input_ids, token_type_ids, attention_mask, speaker_ids, labels = example 
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    speaker_ids = speaker_ids.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, speaker_ids=speaker_ids)
                    #scores = F.sigmoid(logits)
                    loss = self.criterion(logits, labels)

                else:
                    input_ids, token_type_ids, attention_mask, speaker_ids, labels = example 
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    scores = F.sigmoid(logits)
                    loss = self.criterion(scores, labels)

                # back prop & update variables
                if self.args.grad_accumulation_steps > 1:
                    loss = loss / self.args.grad_accumulation_steps
                loss.backward()
                if (step + 1) % self.args.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                train_loss += loss.item()
                global_step += 1
                del input_ids, token_type_ids, attention_mask, labels
                torch.cuda.empty_cache()

                # do eval
                if self.args.eval and step % self.args.eval_steps == 0 or step == len(self.train_dataloader):
               
                    eval_loss = self.eval()
                    train_loss /= self.args.eval_steps
                    if best_eval_loss is None or eval_loss < best_eval_loss:
                        logging.info(f"Saving model with eval_loss = {eval_loss}...\n")
                        torch.save({
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "model": self.model.state_dict(),
                            "train_loss": train_loss,
                            "eval_loss": best_eval_loss
                        }, os.path.join(self.args.output_dir, f"epoch{epoch}_global-step{global_step}"))
                        best_eval_loss = eval_loss
                    logging.info(f"epoch = {epoch}, step = {step}\n")
                    logging.info(f"train_loss = {train_loss}\n")
                    logging.info(f"eval_loss = {eval_loss}\n")
                    writer.add_scalar("Loss/train", train_loss, global_step)
                    writer.add_scalar("Loss/eval", eval_loss, global_step)
                    self.model.train()

    def eval(self):
        self.model.eval()
        eval_loss = 0
        num_batches = 0
        with torch.no_grad():
            for step, example in tqdm(enumerate(self.dev_dataloader), desc="Eval"):
                if self.args.speaker_embeddings:
                    input_ids, token_type_ids, attention_mask, speaker_ids, labels = example

                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    speaker_ids    = speaker_ids.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, speaker_ids=speaker_ids)
                    #scores = F.sigmoid(logits)
                    scores = logits
                    if step <= 5:
                        logging.info(f"batch {step} labels = {labels}")
                        logging.info(f"batch {step} scores = {scores}")
                    loss = self.criterion(logits, labels)
                    eval_loss += loss.item()
                    num_batches += 1
                    del input_ids, token_type_ids, attention_mask, speaker_ids, labels
                    torch.cuda.empty_cache()

                else:
                    #input_ids, token_type_ids, attention_mask, labels = example
                    input_ids, token_type_ids, attention_mask, speaker_ids, labels = example 
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    scores = F.sigmoid(logits)
                    if step <= 5:
                        logging.info(f"batch {step} labels = {labels}")
                        logging.info(f"batch {step} scores = {scores}")
                    loss = self.criterion(scores, labels)
                    eval_loss += loss.item()
                    num_batches += 1
                    del input_ids, token_type_ids, attention_mask, labels
                    torch.cuda.empty_cache()
        return eval_loss / len(self.dev_dataloader)