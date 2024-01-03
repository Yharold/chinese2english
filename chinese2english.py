import torch
import math
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch import nn


class Chinese2English:
    def __init__(self) -> None:
        pass

    def train():
        # 处理数据，得到词表。这里应该分为中文词表和英文词表

        # 将数据分为训练集和测试集

        # 进行训练

        # 循环次数

        # 批量次数

        pass

    def predict():
        pass


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self):
        self.encoder()
        self.decoder()
        pass


class TransformerEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding()
        self.pos_encoding = PositionalEncoding()
        self.blks = nn.Sequential()
        for i in range(3):
            self.blks.add_module(str(i), EncodeLayer())

    def forward(self):
        self.embedding()
        self.pos_encoding()
        for i, blk in enumerate(self.blks):
            blk()
        pass


class TransformerDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding()
        self.pos_encoding = PositionalEncoding()
        self.blks = nn.Sequential()
        for i in range(3):
            self.blks.add_module(str(i), DecodeLayer())

    def forward(self):
        self.embedding()
        self.pos_encoding()
        for i, blk in enumerate(self.blks):
            blk()
        pass


class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        pass


class TransformerDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        pass


class EncodeLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadAttention()
        self.resnorm = ResNorm()
        self.ffn = FFN()

    def forward(self):
        self.attention()
        self.resnorm()
        self.ffn()
        pass


class DecodeLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention1 = MultiHeadAttention()
        self.resnorm1 = ResNorm()
        self.attention2 = MultiHeadAttention()
        self.resnorm2 = ResNorm()
        self.ffn = FFN()

    def forward(self):
        self.attention1()
        self.resnorm1()
        self.attention2()
        self.resnorm2()
        self.ffn()
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_head, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_head = num_head
        self.w_q = nn.Linear(q_dim, q_dim, bias=False)
        self.w_k = nn.Linear(k_dim, k_dim, bias=False)
        self.w_v = nn.Linear(v_dim, v_dim, bias=False)
        self.attentin = DotProductAttention()
        self.concat = Concat()
        self.w_o = nn.Linear(v_dim, v_dim, bias=False)

    def forward(self, q, k, v, valid_lens):
        h_q = self.transpose_qkv(self.w_q(q))
        h_k = self.transpose_qkv(self.w_k(k))
        h_v = self.transpose_qkv(self.w_v(v))
        self.attentin()
        self.concat()
        self.w_o()
        pass

    def transpose_qkv(self, x: torch.Tensor):
        bz, sz, dm = x.shape
        dx = dm / self.num_head
        x = x.reshape(bz, sz, self.num_head, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bz * self.num_head, sz, dx)
        return x

    def transpose_output(self, x: torch.Tensor):
        x = x.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)
        pass


class ResNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        pass


class FFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        pass


class DotProductAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        # matmul(q,v)
        # torch.masked_fill()
        # torch.softmax()
        # matmul(a,v)
        pass


class Concat(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        pass
