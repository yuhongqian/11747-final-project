from transformers import BertForSequenceClassification
import torch
import torch.nn.functional as F


class ContrastiveElectra(BertForSequenceClassification):
    def get_encoding(self, input_ids, token_type_ids, attention_mask):
        return super().forward(input_ids, token_type_ids, attention_mask)

    def forward(self, pos_input_ids, pos_token_type_ids, pos_attention_mask, neg_input_ids, neg_token_type_ids,
                neg_attention_mask):
        pos_logits = self.get_encoding(input_ids=pos_input_ids, token_type_ids=pos_token_type_ids,
                                       attention_mask=pos_attention_mask).logits
        neg_logits = self.get_encoding(input_ids=neg_input_ids, token_type_ids=neg_token_type_ids,
                                       attention_mask=neg_attention_mask).logits
        logit_matrix = torch.cat([pos_logits, neg_logits], dim=1)
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return loss.mean()
