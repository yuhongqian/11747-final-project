from transformers import ElectraEmbeddings
import torch
from torch import nn


class SpeakerAwareEmbeddings(ElectraEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.speaker_embedding = nn.Embedding(2, config.embedding_size)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, speaker_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if speaker_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        speaker_embeddings = self.speaker_embedding(speaker_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + speaker_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings