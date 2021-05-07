import os
import torch
import random
import logging
import numpy as np
import argparse
from torch import nn  
from torch.utils.data import DataLoader
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, ElectraPreTrainedModel, ElectraModel
#from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel)




''' 
NOTE: This is untested code, need to debug a few things before training. 
'''

class SpeakerAwareElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings, and speaker positions."""

    def __init__(self, config, num_speakers=2):
        super().__init__()

        self.num_speakers = num_speakers

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)


        self.speaker_embeddings = nn.Embedding(self.num_speakers, config.embedding_size)   # add embeddings for speaker, how to modify config?

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, speaker_ids = None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if speaker_ids is None:
            speaker_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        speaker_embeddings  = self.speaker_embeddings(speaker_ids)  # add speaker embeddings

        embeddings = inputs_embeds + token_type_embeddings + speaker_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Below model make sure to add a classification layer

class SpeakerAwareElectraModel(ElectraPreTrainedModel):
    def __init__(self, config, num_speakers=2):
        super().__init__(config)

        self.num_speakers = num_speakers
        self.electra      = ElectraModel(config)
        self.embeddings   = SpeakerAwareElectraEmbeddings(config, self.num_speakers)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        speaker_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        
        # INCLUDE SPEAKER IDS HERE
        speaker_ids = speaker_ids.view(-1, speaker_ids.size(-1)) if speaker_ids is not None else None
        speaker_ids = speaker_ids.unsqueeze(-1).repeat([1,1,speaker_ids.size(1)]) if speaker_ids is not None else None
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, speaker_ids=speaker_ids)
        print(inputs_embeds)



        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.electra(
            input_ids=None,
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        #return outputs


config = ElectraConfig.from_pretrained('google/electra-small-discriminator', num_labels=1)   # 1 label for regression
model = SpeakerAwareElectraModel.from_pretrained('google/electra-small-discriminator', config=config)
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

print(model(**inputs))