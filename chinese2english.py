import math
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor, nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
import tokenizers.pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
import json
from IPython import display
import os
import time


class CustomDataset(Dataset):
    def __init__(self, feature, label) -> None:
        super().__init__()
        if len(feature) == len(label):
            self.feature = feature
            self.label = label
            self.length = len(feature)
        else:
            print("feature is not  equal to label")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.feature[index], self.label[index])


def custom_tokenizer(
    data, vz, sz, language="english", special_tok=["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    if language == "chinese":
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
    else:
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    trainer = BpeTrainer(
        vocab_size=vz,
        special_tokens=special_tok,
    )
    tokenizer.decoder = BPEDecoder()
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", 1),
            ("[EOS]", 2),
            ("[PAD]", 3),
        ],
    )
    if type(data) == "str":
        tokenizer.train(data)
    if hasattr(data, "__iter__"):
        tokenizer.train_from_iterator(data, trainer)
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=sz)
    tokenizer.enable_truncation(max_length=sz)
    return tokenizer


def load_tokenizer(vocab_size, sz):
    feature, label = read_data()
    if "enc_tokenizer.json" in os.listdir("./datasets"):
        enc_tokenizer = Tokenizer.from_file(".\datasets\enc_tokenizer.json")
    else:
        print("train enc_tokenizer")
        enc_tokenizer = custom_tokenizer(feature, vocab_size, sz, language="chinese")
        enc_tokenizer.save(".\datasets\enc_tokenizer.json")
    if "dec_tokenizer.json" in os.listdir("./datasets"):
        dec_tokenizer = Tokenizer.from_file(".\datasets\dec_tokenizer.json")
    else:
        print("train dec_tokenizer")
        dec_tokenizer = custom_tokenizer(label, vocab_size, sz, language="english")
        dec_tokenizer.save(".\datasets\dec_tokenizer.json")
    return feature, label, enc_tokenizer, dec_tokenizer


def read_data(filepath="datasets\\translation2019zh_valid.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    chinese_data = []
    english_data = []
    for line in data:
        chinese_data.append(line["chinese"])
        english_data.append(line["english"])
    return chinese_data, english_data


class Chinese2English:
    def __init__(self) -> None:
        pass

    # def train(self, net, epchos, bz, vz, sz, lr, device="cpu"):
    #     torch.autograd.set_detect_anomaly(True)
    #     # 处理数据，得到词表。这里应该分为中文词表和英文词表
    #     feature, label, enc_tokenizer, dec_tokenizer = load_tokenizer(vz, sz)
    #     # 将数据分为训练集和测试集
    #     data_size = len(feature)
    #     train_data = CustomDataset(
    #         feature[0 : int(data_size * 0.8)], label[0 : int(data_size * 0.8)]
    #     )
    #     test_data = CustomDataset(
    #         feature[int(data_size * 0.8) :], label[int(data_size * 0.8) :]
    #     )
    #     # 得到DataLoader
    #     train_dataloader = DataLoader(train_data, bz, shuffle=True)
    #     test_dataloader = DataLoader(test_data, bz, shuffle=True)

    #     # 初始化模型
    #     def init_net(m):
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         if isinstance(m, nn.Embedding):
    #             nn.init.xavier_uniform_(m.weight)

    #     net.apply(init_net)
    #     loss = CustomLoss()
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #     # 进行训练
    #     net.train()
    #     loss_record = []
    #     for epcho in range(epchos):
    #         metric = [0.0, 0.0, 0.0]
    #         start_time = time.time()
    #         for iter in train_dataloader:
    #             X, Y = iter[0], iter[1]
    #             X = enc_tokenizer.encode_batch(X)
    #             enc_valid = torch.tensor([sum(x.attention_mask) for x in X])
    #             Y = dec_tokenizer.encode_batch(Y)
    #             Y_mask = torch.tensor([y.attention_mask for y in Y])
    #             dec_valid = torch.tensor([sum(y) for y in Y_mask])
    #             X = torch.tensor([x.ids for x in X])
    #             Y = torch.tensor([y.ids for y in Y])
    #             dec_output = net(X, Y, enc_valid, dec_valid)
    #             l = loss(dec_output, Y, Y_mask)
    #             # 清零梯度
    #             optimizer.zero_grad()
    #             # 计算梯度
    #             l.backward()
    #             # grad_clipping(net, 1)
    #             # 更新梯度
    #             optimizer.step()
    #             torch.cuda.empty_cache()
    #             with torch.no_grad():
    #                 metric[0] += l
    #                 metric[1] += dec_valid.sum()
    #                 metric[2] += time.time() - start_time
    #             loss_record.append(metric[0] / metric[1])
    #             print(loss_record[-1])
    #         if (epcho + 1) % 10 == 0:
    #             filename = "./datasets/net/transformer-" + str(epcho)
    #             torch.save(net.state_dict(), filename)
    #     print(
    #         f"loss {loss_record[epcho]:.3f}, {metric[1] / metric[2]:.1f} "
    #         f"tokens/sec on {str(device)}"
    #     )
    #     torch.save(net.state_dict(), "./datasets/net/transformer-all_done")

    # 输入模型，处理好的包含了feture,label的dataloader,num_epochs,lr
    def train(model: nn.Module, dataloader: DataLoader, num_epochs, lr):
        def init_model(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        model.apply(init_model)
        loss = CustomLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for iter in dataloader:
                X, Y, X_valid, Y_valid, Y_mask = (
                    iter[0],
                    iter[1],
                    iter[2],
                    iter[3],
                    iter[4],
                )
                pred = model(X, Y, X_valid, Y_valid)
                l = loss(pred, Y, Y_mask)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

    def predict():
        pass


class CustomLoss(nn.CrossEntropyLoss):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, Y_mask: torch.Tensor):
        self.reduction = "none"
        unweighted_loss = super(CustomLoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * Y_mask).mean(dim=1).sum()
        return weighted_loss


def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y, enc_valid, dec_valid):
        enc_output = self.encoder(X, enc_valid)
        return self.decoder(Y, dec_valid, enc_output, enc_valid)


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
        # print(self.__class__.__name__)
        X = self.embedding(X) * math.sqrt(self.dm)
        X = self.pos_encoding(X)
        self.attention_weight = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weight[i] = blk.attention.attention.weights
        return X


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
        self.weights = [[None] * num_layer] * 2

    def forward(self, X, dec_valid, enc_output, enc_valid):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.dm))
        if dec_valid is not None and dec_valid.dim() == 1:
            bz, sz, _ = X.shape
            dec_valid = torch.repeat_interleave(dec_valid, sz).reshape(bz, -1)
            dec_valid = torch.min(dec_valid, torch.arange(1, sz + 1).repeat(bz, 1))
        enc_info = (enc_output, enc_valid)
        dec_input = (X, self.key_value, dec_valid)
        for i, blk in enumerate(self.blks):
            dec_input, enc_info = blk(dec_input, enc_info)
            self.weights[0][i] = blk.attention1.attention.weights
            self.weights[1][i] = blk.attention2.attention.weights
        return self.linear(dec_input[0])


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
        # print(self.__class__.__name__)
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
        # print(self.__class__.__name__)
        Y = self.resnorm1(X, self.attention(X, X, X, valid_lens))
        return self.resnorm2(Y, self.ffn(Y))


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
        # print(self.__class__.__name__)
        Q, KV, dec_valid = dec_input[0], dec_input[1], dec_input[2]
        enc_output, enc_valid = enc_info[0], enc_info[1]
        if KV[self.idx] is None:
            KV[self.idx] = Q
        else:
            KV[self.idx] = torch.cat((KV[self.idx], Q), dim=1)
        Y = self.resnorm1(Q, self.attention1(Q, KV[self.idx], KV[self.idx], dec_valid))
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
        # print(self.__class__.__name__)
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
        # print(self.__class__.__name__)
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
        # print(self.__class__.__name__)
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
        # print(self.__class__.__name__)
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
