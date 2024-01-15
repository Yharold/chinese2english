import math
import random
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


def custom_data(
    feature_vocab_size,
    label_vocab_size,
    feature_seq_size,
    label_seq_size,
    feature_filename="feature_tokenizer",
    label_filname="label_tokenizer",
    XY_filename="XY.pt",
):
    # 读取原始数据
    feature, label = read_data()
    # 创建分词器
    feature_tokenizer = custom_tokenizer(
        feature, feature_vocab_size, feature_seq_size, feature_filename
    )
    label_tokenizer = custom_tokenizer(
        label, label_vocab_size, label_seq_size, label_filname
    )
    # 将所有文本序列转为数字序列，得到数字序列和掩码
    XY_filepath = "datasets/" + XY_filename
    # 如果已经计算过并保存为文件，那直接从文件读取，否则重新计算
    if XY_filename not in os.listdir("datasets"):
        X, X_valid, X_mask = text_seq2num_seq(feature, feature_tokenizer)
        Y, Y_valid, Y_mask = text_seq2num_seq(label, label_tokenizer)
        torch.save([X, X_valid, X_mask, Y, Y_valid, Y_mask], XY_filepath)
    else:
        X, X_valid, X_mask, Y, Y_valid, Y_mask = torch.load(XY_filepath)
    return X, X_valid, X_mask, Y, Y_valid, Y_mask, feature, label


def custom_train_test_data():
    XY_filepath = "datasets/XY.pt"
    X, X_valid, X_mask, Y, Y_valid, Y_mask = torch.load(XY_filepath)
    length = X.shape[0]
    print(length)
    i = int(length * 0.8)
    print(i)
    train_X, train_X_valid, train_X_mask, train_Y, train_Y_valid, train_Y_mask = (
        X[0:i],
        X_valid[0:i],
        X_mask[0:i],
        Y[0:i],
        Y_valid[0:i],
        Y_mask[0:i],
    )
    torch.save(
        [train_X, train_X_valid, train_X_mask, train_Y, train_Y_valid, train_Y_mask],
        "datasets/train_XY.pt",
    )
    test_X, test_X_valid, test_X_mask, test_Y, test_Y_valid, test_Y_mask = (
        X[i:],
        X_valid[i:],
        X_mask[i:],
        Y[i:],
        Y_valid[i:],
        Y_mask[i:],
    )
    torch.save(
        [test_X, test_X_valid, test_X_mask, test_Y, test_Y_valid, test_Y_mask],
        "datasets/test_XY.pt",
    )


def custom_tokenizer(data, vocab_size, seq_size, filename):
    file_path = "datasets/" + filename
    if filename not in os.listdir("datasets"):
        special_tokens = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        feature_trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )
        tokenizer.decoder = BPEDecoder()
        tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=seq_size)
        tokenizer.enable_truncation(max_length=seq_size)
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)]
        )
        tokenizer.train_from_iterator(data, feature_trainer)
        tokenizer.save(file_path)
    else:
        tokenizer = Tokenizer.from_file(file_path)
    return tokenizer


def text_seq2num_seq(data, tokenizer: Tokenizer):
    X = []
    X_valid = []
    X_mask = []
    for iter in data:
        tmp = tokenizer.encode(iter)
        X.append(tmp.ids)
        X_mask.append(tmp.attention_mask)
        X_valid.append([sum(tmp.attention_mask) for _ in range(len(tmp.ids))])
    return torch.tensor(X), torch.tensor(X_valid), torch.tensor(X_mask)


def read_data(filepath="datasets/translation2019zh_valid.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    chinese_data = []
    english_data = []
    for line in data:
        chinese_data.append(line["chinese"])
        english_data.append(line["english"])
    return chinese_data, english_data


def init_model(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


class CustomDataset(Dataset):
    def __init__(self, X, X_valid, X_mask, Y, Y_valid, Y_mask) -> None:
        super().__init__()
        self.length = X.shape[0]
        self.X = X
        self.Y = Y
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_mask = X_mask
        self.Y_mask = Y_mask

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.X[index],
            self.X_valid[index],
            self.X_mask[index],
            self.Y[index],
            self.Y_valid[index],
            self.Y_mask[index],
        )


class Chinese2English:
    def __init__(self) -> None:
        pass

    # 输入模型，处理好的包含了feture,label的dataloader,num_epochs,lr
    def train(self, model: nn.Module, dataloader: DataLoader, num_epochs, lr, device):
        model.apply(init_model)
        model.to(device)
        loss = CustomLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        each_scores = []
        all_time = 0.0
        for epoch in range(num_epochs):
            scores = [0.0, 0.0, 0.0]
            start = time.time()
            for iter in dataloader:
                # 大小应该都是(bz,sz)
                X, X_valid, _, Y, _, Y_mask = [x.to(device) for x in iter]
                Y_valid = torch.arange(1, Y.shape[1] + 1, device=device).expand(Y.shape)
                pred, _ = model(X, Y, X_valid, Y_valid)
                l = loss(pred, Y, Y_mask)
                optimizer.zero_grad()
                l.backward()
                grad_clipping(model, 1)
                optimizer.step()
                with torch.no_grad():
                    scores[0] += l.item()
                    scores[1] += Y_mask.sum().sum().item()
                    scores[2] = scores[1] / (time.time() - start)
            each_scores.append(scores[0] / scores[1])
            all_time += time.time() - start
            if (epoch + 1) % 10 == 0:
                torch.save(
                    model.state_dict(), "datasets/model/model_" + str(epoch) + ".pt"
                )
                torch.save(each_scores, "datasets/scroes.pt")
                print(
                    epoch + 1,
                    f"loss:, {each_scores[-1]:.3f}, {scores[2]:.3f}token/second on {device}",
                    f"processing time: {all_time} second",
                    f"{(epoch+1)/num_epochs*100:.1f}%,"
                    f"predict to need for {all_time*(num_epochs/(epoch+1)-1):.1f}",
                )
            torch.save(model.state_dict(), "datasets/model/model_all.pt")
            torch.save(each_scores, "datasets/scroes.pt")

    def predict(model: nn.Module, dataloader: DataLoader):
        pass


class CustomLoss(nn.CrossEntropyLoss):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, Y_mask: torch.Tensor):
        self.reduction = "none"
        unweighted_loss = super(CustomLoss, self).forward(pred.permute(0, 2, 1), label)
        # weighted_loss = (unweighted_loss * Y_mask).mean(dim=1).sum()
        weighted_loss = (
            (unweighted_loss * Y_mask).sum(dim=1) / (Y_mask.sum(dim=1))
        ).sum()
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
        key_value = [None] * self.decoder.num_layer
        dec_input = (Y, key_value, dec_valid)
        enc_info = (enc_output, enc_valid)
        return self.decoder(dec_input, enc_info)


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

    def forward(self, X, valid):
        # print(self.__class__.__name__)
        if valid.dim() == 1:
            valid = valid.expand(X.shape[0], X.shape[1])
        X = self.embedding(X) * math.sqrt(self.dm)
        X = self.pos_encoding(X)
        self.attention_weight = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid)
            self.attention_weight[i] = blk.attention.attention.weights
        return X


class TransformerDecoder(nn.Module):
    def __init__(
        self, voca_size, dm, num_hidden, num_head, dropout, num_layer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dm = dm
        self.num_layer = num_layer
        self.embedding = nn.Embedding(voca_size, dm)
        self.pos_encoding = PositionalEncoding(dm, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layer):
            self.blks.add_module(
                str(i), DecodeLayer(i, dm, num_hidden, num_head, dropout)
            )
        self.linear = nn.Linear(dm, voca_size)
        self.weights = [[None] * num_layer] * 2

    def forward(self, dec_input, enc_info):
        X, key_value, dec_valid = dec_input
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.dm))
        dec_input = (X, key_value, dec_valid)
        for i, blk in enumerate(self.blks):
            dec_input, enc_info = blk(dec_input, enc_info)
            self.weights[0][i] = blk.attention1.attention.weights
            self.weights[1][i] = blk.attention2.attention.weights
        Y = self.linear(dec_input[0])
        return Y, key_value


class PositionalEncoding(nn.Module):
    def __init__(self, dm, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.dm = dm

    def forward(self, X: Tensor):
        # print(self.__class__.__name__)
        max_len = 1000
        self.P = torch.zeros((1, max_len, self.dm), device=X.device)
        i = torch.arange(max_len, dtype=torch.float32, device=X.device).reshape(-1, 1)
        j = torch.arange(0, self.dm, 2, dtype=torch.float32, device=X.device)
        tmp = i / torch.pow(10000, j / self.dm)
        self.P[:, :, 0::2] = torch.sin(tmp)
        self.P[:, :, 1::2] = torch.cos(tmp)
        return self.dropout(X + self.P[:, : X.shape[1], :])


class EncodeLayer(nn.Module):
    def __init__(self, dm, num_hidden, num_head, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadAttention(dm, dm, dm, num_head, dropout)
        self.resnorm1 = ResNorm(dm, dropout)
        self.ffn = FFN(dm, num_hidden)
        self.resnorm2 = ResNorm(dm, dropout)

    def forward(self, X, valid=None):
        # print(self.__class__.__name__)
        Y = self.resnorm1(X, self.attention(X, X, X, valid))
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

    def forward(self, q, k, v, valid):
        # print(self.__class__.__name__)
        # h_q等大小是(bz*num_head,sz,dm/num_head),valid大小(bz,sz)
        h_q = self.transpose_qkv(self.w_q(q))
        h_k = self.transpose_qkv(self.w_k(k))
        h_v = self.transpose_qkv(self.w_v(v))
        # 所以valid也要同步扩大
        if valid is not None:
            new_valid = (
                valid.unsqueeze(1)
                .expand(q.shape[0], self.num_head, q.shape[1])
                .reshape(-1, q.shape[1])
            )
        output = self.attention(h_q, h_k, h_v, new_valid)
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
        y = x.reshape(-1, self.num_head, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)
        return y.reshape(y.shape[0], y.shape[1], -1)


class ResNorm(nn.Module):
    def __init__(self, dm, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(dm)

    def forward(self, X, Y):
        # print(self.__class__.__name__)
        # 先执行dropout，再执行残差连接
        # 再执行layernorm
        return self.layernorm(self.dropout(Y) + X)


class FFN(nn.Module):
    def __init__(self, dm, num_hidden, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(dm, num_hidden, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, dm, bias=True)

    def forward(self, X):
        # print(self.__class__.__name__)
        return self.linear2(self.relu(self.linear1(X)))


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
        valid,
    ):
        # print(self.__class__.__name__)
        # 总公式是 result = softmax(q*v_T/sqrt(dm))v,
        # 计算q*v_T/sqrt(dm),v_T是v的转置，这里的dm我选择的是k，其实q，k，v都一样
        scores = torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(k.shape[-1])
        # 计算掩码并进行掩码操作,valid:(bz,sz)
        if valid is not None:
            mask = torch.arange(1, q.shape[1] + 1, device=q.device).expand(
                (q.shape[0], q.shape[1], q.shape[1])
            ) > valid.unsqueeze(2)
            mask_scores = torch.masked_fill(scores, mask, 1e-6)
            # 在最后一个维度执行softmax函数，得到的就是我们所说的注意力权重
            self.weights = torch.softmax(mask_scores, dim=-1)
        else:
            self.weights = torch.softmax(scores, dim=-1)
        # 计算权值与v的乘积，权值要先执行dropout
        return torch.bmm(self.dropout(self.weights), v)
