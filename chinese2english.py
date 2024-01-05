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

    def forward(self, X, Y, enc_valid, dec_valid):
        enc_output = self.encoder(X, enc_valid)
        return self.decoder(Y, enc_output, enc_valid, dec_valid)


class TransformerEncoder(nn.Module):
    def __init__(
        self, voca_size, dm, num_hidden, num_head, dropout, num_layer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dm = dm
        self.embedding = nn.Embedding(voca_size, dm)
        self.pos_encoding = PositionalEncoding(dm, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layer):
            self.blks.add_module(str(i), EncodeLayer(dm, num_hidden, num_head, dropout))

    def forward(self, X, valid_lens):
        print(self.__class__.__name__)
        X = self.embedding(X) * math.sqrt(self.dm)
        X = self.pos_encoding(X)
        self.attention_weight = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weight[i] = blk.attention.attention.weights
        return X


# class TransformerDecoder(nn.Module):
#     def __init__(
#         self, voca_size, dm, num_hidden, num_head, dropout, num_layer, *args, **kwargs
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.dm = dm
#         self.embedding = nn.Embedding(voca_size, dm)
#         self.pos_encoding = PositionalEncoding(dm, dropout)
#         self.blks = nn.Sequential()
#         for i in range(num_layer):
#             self.blks.add_module(
#                 str(i), DecodeLayer(i, dm, num_hidden, num_head, dropout)
#             )
#         self.linear = nn.Linear(dm, voca_size)
#         self.key_value = [None] * num_layer

#     def forward(self, X, enc_output, enc_valid, dec_valid=None):
#         X = self.embedding(X) * math.sqrt(self.dm)
#         X = self.pos_encoding(X)
#         self.attention_weight1 = [None] * len(self.blks)
#         self.attention_weight2 = [None] * len(self.blks)
#         for i, blk in enumerate(self.blks):
#             if self.training:
#                 X = blk(X, enc_output, enc_valid, dec_valid)
#                 self.attention_weight1[i] = blk.attention1.attention.weights
#                 self.attention_weight2[i] = blk.attention2.attention.weights
#             else:
#                 X, self.key_value = blk(
#                     X, enc_output, enc_valid, dec_valid, self.key_value
#                 )
#         return self.linear(X)


class TransformerDecoder(nn.Module):
    def __init__(
        self, voca_size, dm, num_hidden, num_head, dropout, num_layer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dm = dm
        self.embedding = nn.Embedding(voca_size, dm)
        self.pos_encoding = PositionalEncoding(dm, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layer):
            self.blks.add_module(
                str(i), DecodeLayer(i, dm, num_hidden, num_head, dropout)
            )
        self.linear = nn.Linear(dm, voca_size)
        self.key_value = [None] * num_layer

    def forward(self, X, enc_output, enc_valid, dec_valid=None):
        X = self.embedding(X) * math.sqrt(self.dm)
        X = self.pos_encoding(X)
        self.attention_weight1 = [None] * len(self.blks)
        self.attention_weight2 = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            if self.training:
                X = blk(X, enc_output, enc_valid, dec_valid)
                self.attention_weight1[i] = blk.attention1.attention.weights
                self.attention_weight2[i] = blk.attention2.attention.weights
            else:
                X, self.key_value = blk(
                    X, enc_output, enc_valid, dec_valid, self.key_value
                )
        return self.linear(X)


class PositionalEncoding(nn.Module):
    def __init__(self, dm, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        max_len = 1000
        self.P = torch.zeros((1, max_len, dm))
        i = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        j = torch.arange(0, dm, 2, dtype=torch.float32)
        X = i / torch.pow(10000, j / dm)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        print(self.__class__.__name__)
        Y = X + self.P[:, : X.shape[1], :]
        return self.dropout(Y)


class EncodeLayer(nn.Module):
    def __init__(self, dm, num_hidden, num_head, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadAttention(dm, dm, dm, num_head, dropout)
        self.resnorm1 = ResNorm(dm, dropout)
        self.ffn = FFN(dm, num_hidden)
        self.resnorm2 = ResNorm(dm, dropout)

    def forward(self, X, valid_lens=None):
        print(self.__class__.__name__)
        Y = self.resnorm1(X, self.attention(X, X, X, valid_lens))
        return self.resnorm2(Y, self.ffn(Y))


# class DecodeLayer(nn.Module):
#     def __init__(self, idx, dm, num_hidden, num_head, dropout, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.idx = idx
#         self.attention1 = MultiHeadAttention(dm, dm, dm, num_head, dropout)
#         self.resnorm1 = ResNorm(dm, dropout)
#         self.attention2 = MultiHeadAttention(dm, dm, dm, num_head, dropout)
#         self.resnorm2 = ResNorm(dm, dropout)
#         self.ffn = FFN(dm, num_hidden)
#         self.resnorm3 = ResNorm(dm, dropout)

#     def forward(self, X, enc_output, enc_valid, dec_valid, key_value=None):
#         print(self.__class__.__name__)
#         if self.training:
#             bz, sz, _ = X.shape
#             dec_valid = torch.repeat_interleave(dec_valid, sz).reshape(bz, -1)
#             tmp = torch.arange(1, sz + 1).repeat(bz, 1)
#             dec_valid = torch.min(dec_valid, tmp)

#             Y = self.resnorm1(X, self.attention1(X, X, X, dec_valid))
#             Y2 = self.resnorm2(Y, self.attention2(Y, enc_output, enc_output, enc_valid))
#             return self.resnorm3(Y2, self.ffn(Y2))
#         else:
#             dec_valid = None
#             if key_value[self.idx] is None:
#                 key_value[self.idx] = X
#             else:
#                 key_value[self.idx] = torch.cat((key_value[self.idx], X), dim=1)
#             Y = self.resnorm1(
#                 X,
#                 self.attention1(X, key_value[self.idx], key_value[self.idx], dec_valid),
#             )
#             Y2 = self.resnorm2(Y, self.attention2(Y, enc_output, enc_output, enc_valid))
#             return self.resnorm3(Y2, self.ffn(Y2)), key_value


class DecodeLayer(nn.Module):
    def __init__(self, idx, dm, num_hidden, num_head, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.idx = idx
        self.attention1 = MultiHeadAttention(dm, dm, dm, num_head, dropout)
        self.resnorm1 = ResNorm(dm, dropout)
        self.attention2 = MultiHeadAttention(dm, dm, dm, num_head, dropout)
        self.resnorm2 = ResNorm(dm, dropout)
        self.ffn = FFN(dm, num_hidden)
        self.resnorm3 = ResNorm(dm, dropout)

    def forward(self, dec_input, enc_info):
        print(self.__class__.__name__)
        Q, KV, dec_valid = dec_input[0], dec_input[1], dec_input[2]
        enc_output, enc_valid = enc_info[0], enc_info[1]
        if KV[self.idx] is None:
            KV[self.idx] = Q
        else:
            KV[self.idx] = torch.cat((KV[self.idx], Q), dim=-1)
        Y = self.resnorm1(Q, self.attention1(Q, KV, KV, dec_valid))
        Y2 = self.resnorm2(Y, self.attention2(Y, enc_output, enc_output, enc_valid))
        Y3 = self.resnorm3(Y2, self.ffn(Y2))
        return (Y3, KV, dec_valid), enc_info


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_head, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_head = num_head
        self.w_q = nn.Linear(q_dim, q_dim, bias=False)
        self.w_k = nn.Linear(k_dim, k_dim, bias=False)
        self.w_v = nn.Linear(v_dim, v_dim, bias=False)
        self.attention = DotProductAttention(dropout)
        self.w_o = nn.Linear(v_dim, v_dim, bias=False)

    def forward(self, q, k, v, valid_lens=None):
        print(self.__class__.__name__)
        h_q = self.transpose_qkv(self.w_q(q))
        h_k = self.transpose_qkv(self.w_k(k))
        h_v = self.transpose_qkv(self.w_v(v))
        # h_q等大小是(bz*num_head,sz,dm/num_head),所以valid_lens也要同步扩大
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_head, dim=0)
        output = self.attention(h_q, h_k, h_v, valid_lens)
        output_concat = self.transpose_output(output)
        return self.w_o(output_concat)

    # 输入x:(bz,sz,dm),输出:(bz*num_head,sz,dm/num_head)
    def transpose_qkv(self, x: torch.Tensor):
        bz, sz, _ = x.shape
        return (
            x.reshape(bz, sz, self.num_head, -1)
            .permute(0, 2, 1, 3)
            .reshape(bz * self.num_head, sz, -1)
        )

    # 输入x:(bz*num_head,sz,dm/num_head),输出:(bz,sz,dm)
    def transpose_output(self, x: torch.Tensor):
        x = x.reshape(-1, self.num_head, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


class ResNorm(nn.Module):
    def __init__(self, dm, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(dm)

    def forward(self, X, Y):
        print(self.__class__.__name__)
        # 先执行dropout，再执行残差连接
        Y = self.dropout(Y) + X
        # 再执行layernorm
        return self.layernorm(Y)


class FFN(nn.Module):
    def __init__(self, dm, num_hidden, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(dm, num_hidden, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, dm, bias=True)

    def forward(self, X):
        print(self.__class__.__name__)
        Y = self.linear1(X)
        Y = self.relu(Y)
        Y = self.linear2(Y)
        return Y


class DotProductAttention(nn.Module):
    def __init__(self, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)

    # 多头注意力中q,k,v都是(bz*num_head,sz,dm/num_head)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_lens=None,
    ):
        print(self.__class__.__name__)
        # 总公式是 result = softmax(q*v_T/sqrt(dm))v,
        # 计算q*v_T/sqrt(dm),v_T是v的转置，这里的dm我选择的是k，其实q，k，v都一样
        scores = torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(k.shape[-1])
        # 计算掩码并进行掩码操作
        if valid_lens is not None:
            # 编码器输入和输出的有效长度valid_len是维度为1的张量，大小是bz
            if valid_lens.dim() == 1:
                valid_lens = valid_lens.reshape(scores.shape[0], 1, 1)
            # 解码器的输入有效长度valid_len是维度为2的张量，大小是(bz,sz)
            else:
                valid_lens = valid_lens.reshape(scores.shape[0], scores.shape[1], -1)
            mask = torch.arange(scores.shape[-1], dtype=torch.int16).expand(
                scores.shape
            )
            mask = mask >= valid_lens
            scores = torch.masked_fill(scores, mask, 1e-6)
        # 在最后一个维度执行softmax函数，得到的就是我们所说的注意力权重
        self.weights = torch.softmax(scores, dim=-1)
        # 计算权值与v的乘积，权值要先执行dropout
        return torch.bmm(self.dropout(self.weights), v)


class Concat(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        print(self.__class__.__name__)
        pass
