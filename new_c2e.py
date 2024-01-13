import math
import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
    ):
        enc_output = self.encoder()
        dec_output = self.decoder()
        return dec_output


class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size, dm, num_heads, num_hidden, num_layer, dropout, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, dm)
        self.pos_encoding = PositionalEncoding(dropout)
        self.blks = nn.Sequential()
        for i in range(num_layer):
            self.blks.add_module(str(i), EncoderBlock())

    def forward(self):
        pass


class PositionalEncoding:
    pass


class EncoderBlock:
    pass


class ScaleDotProductAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
