import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


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
        writer = SummaryWriter()
        optimizer = Adam(params=self.model.parameters(), lr=self.args.lr, eps=1e-8, weight_decay=self.args.weight_decay)
        global_step = 0
        best_eval_loss = None
        for epoch in tqdm(range(self.args.epochs), desc="Epoch"):
            train_loss = 0
            for step, example in tqdm(enumerate(self.train_dataloader), desc="Train"):
                # get outputs
                input_ids, token_type_ids, attention_mask, labels = example
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
                    optimizer.zero_grad()
                train_loss += loss.item()
                global_step += 1
                del input_ids, token_type_ids, attention_mask, labels
                torch.cuda.empty_cache()

                # do eval
                if self.args.eval and (step + 1) % self.args.eval_steps == 0:
                    eval_loss = self.eval()
                    train_loss /= self.args.eval_steps
                    if best_eval_loss is None or eval_loss < best_eval_loss:
                        logging.info(f"Saving model with eval_loss = {best_eval_loss}...\n")
                        torch.save({
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
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
            for step, example in tqdm(enumerate(self.train_dataloader), desc="Eval"):
                input_ids, token_type_ids, attention_mask, labels = example
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                logits = outputs.logits
                scores = F.sigmoid(logits)
                loss = self.criterion(scores, labels)
                eval_loss += loss.item()
                num_batches += 1
                del input_ids, token_type_ids, attention_mask, labels
                torch.cuda.empty_cache()
        return eval_loss