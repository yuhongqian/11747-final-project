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


class Trainer:
    def __init__(self, args, model, device, train_dataloader, dev_dataloader):
        self.args = args
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.criterion = nn.MSELoss(reduction="mean")

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
                # get outputs
                if self.args.contrastive:
                    pos_input_ids, pos_token_type_ids, pos_attention_mask, neg_input_ids, neg_token_type_ids, \
                    neg_attention_mask = example
                    if self.args.train_batch_size == 1:
                        pos_input_ids = pos_input_ids.unsqueeze(0)
                        pos_token_type_ids = pos_token_type_ids.unsqueeze(0)
                        pos_attention_mask = pos_attention_mask.unsqueeze(0)
                        neg_input_ids = neg_input_ids.unsqueeze(0)
                        neg_token_type_ids = neg_token_type_ids.unsqueeze(0)
                        neg_attention_mask = neg_attention_mask.unsqueeze(0)
                    pos_input_ids = pos_input_ids.to(self.device)
                    pos_token_type_ids = pos_token_type_ids.to(self.device)
                    pos_attention_mask = pos_attention_mask.to(self.device)

                    # logging.info(f"{pos_input_ids.shape} {pos_token_type_ids.shape}, {pos_attention_mask.shape}")
                    pos_logits = self.model(input_ids=pos_input_ids, token_type_ids=pos_token_type_ids,
                                            attention_mask=pos_attention_mask).logits
                    del pos_input_ids, pos_token_type_ids, pos_attention_mask
                    torch.cuda.empty_cache()

                    neg_input_ids = neg_input_ids.to(self.device)
                    neg_token_type_ids = neg_token_type_ids.to(self.device)
                    neg_attention_mask = neg_attention_mask.to(self.device)
                    neg_logits = self.model(input_ids=neg_input_ids, token_type_ids=neg_token_type_ids,
                                            attention_mask=neg_attention_mask).logits
                    del neg_input_ids, neg_token_type_ids, neg_attention_mask
                    torch.cuda.empty_cache()

                    logit_matrix = torch.cat([pos_logits, neg_logits], dim=1)
                    lsm = F.log_softmax(logit_matrix, dim=1)
                    loss = -1.0 * lsm[:, 0].mean()
                else:
                    input_ids, token_type_ids, attention_mask, labels = example
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    scores = F.sigmoid(logits)
                    loss = self.criterion(scores, labels)
                    del input_ids, token_type_ids, attention_mask, labels
                    torch.cuda.empty_cache()
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

                # do eval
                if self.args.eval and (step + 1) % self.args.eval_steps == 0 or step == len(self.train_dataloader):
                # if self.args.eval and (step + 1) % self.args.eval_steps == 0 or step == len(self.train_dataloader):
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
                if self.args.contrastive:
                    pos_input_ids, pos_token_type_ids, pos_attention_mask, neg_input_ids, neg_token_type_ids, \
                    neg_attention_mask = example
                    if self.args.train_batch_size == 1:
                        pos_input_ids = pos_input_ids.unsqueeze(0)
                        pos_token_type_ids = pos_token_type_ids.unsqueeze(0)
                        pos_attention_mask = pos_attention_mask.unsqueeze(0)
                        neg_input_ids = neg_input_ids.unsqueeze(0)
                        neg_token_type_ids = neg_token_type_ids.unsqueeze(0)
                        neg_attention_mask = neg_attention_mask.unsqueeze(0)

                    pos_input_ids = pos_input_ids.to(self.device)
                    pos_token_type_ids = pos_token_type_ids.to(self.device)
                    pos_attention_mask = pos_attention_mask.to(self.device)
                    pos_logits = self.model(input_ids=pos_input_ids, token_type_ids=pos_token_type_ids,
                                                   attention_mask=pos_attention_mask).logits
                    del pos_input_ids, pos_token_type_ids, pos_attention_mask
                    torch.cuda.empty_cache()

                    neg_input_ids = neg_input_ids.to(self.device)
                    neg_token_type_ids = neg_token_type_ids.to(self.device)
                    neg_attention_mask = neg_attention_mask.to(self.device)
                    neg_logits = self.model(input_ids=neg_input_ids, token_type_ids=neg_token_type_ids,
                                                   attention_mask=neg_attention_mask).logits
                    del neg_input_ids, neg_token_type_ids, neg_attention_mask
                    torch.cuda.empty_cache()

                    logit_matrix = torch.cat([pos_logits, neg_logits], dim=1)
                    lsm = F.log_softmax(logit_matrix, dim=1)
                    loss = -1.0 * lsm[:, 0].mean()
                else:
                    input_ids, token_type_ids, attention_mask, labels = example
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    scores = F.sigmoid(logits)
                    loss = self.criterion(scores, labels)
                    del input_ids, token_type_ids, attention_mask, labels
                    torch.cuda.empty_cache()

                eval_loss += loss.item()
                num_batches += 1
        return eval_loss / len(self.dev_dataloader)


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
                input_ids, token_type_ids, attention_mask, _ = example
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                logits = outputs.logits
                scores = logits
                # scores = F.sigmoid(logits)
                del input_ids, token_type_ids, attention_mask
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







