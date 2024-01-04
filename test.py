from chinese2english import *
import numpy as np


def test_MHA():
    bz = 1
    sz = 10
    dm = 15
    num_head = 3
    dropout = 0.1
    shape = (bz, sz, dm)
    q = torch.randn((1, 1, dm))
    k = torch.randn(shape)
    v = torch.randn(shape)
    valid_lens = [7, 9]
    valid_lens = torch.tensor(valid_lens)
    valid_lens = None
    mha = MultiHeadAttention(dm, dm, dm, num_head, dropout)
    result = mha(q, k, v, valid_lens)
    print(result.shape)


def test_ResNorm():
    bz = 1
    sz = 10
    dm = 15
    dropout = 0
    shape = (bz, sz, dm)
    x = torch.randint(1, 10, shape, dtype=torch.float32)
    y = torch.randint(1, 10, shape, dtype=torch.float32) * 2
    resnorm = ResNorm(dm, dropout)
    result = resnorm(x, y)
    r = result[0, :, 1]
    mean_value = np.mean(r.detach().numpy())
    variance_value = np.var(r.detach().numpy())
    print(mean_value, variance_value)


def test_FFN():
    X = torch.randn((2, 10, 15))
    ffn = FFN(15, 60)
    result = ffn(X)
    print(result.shape)


def test_EncoderLayer():
    bz = 2
    sz = 10
    dm = 15
    num_hidden = 60
    num_head = 5
    dropout = 0.1
    shape = (bz, sz, dm)
    X = torch.randn(shape)
    valid_lens = [7, 9]
    valid_lens = torch.tensor(valid_lens)
    encoderlayer = EncodeLayer(dm, num_hidden, num_head, dropout)
    result = encoderlayer(X, valid_lens)
    print(result.shape)
    print(encoderlayer.named_parameters)


def test_PositionalEncoding():
    dm = 16
    X = torch.randn((1, 10, 16))
    dropout = 0.1
    pe = PositionalEncoding(dm, dropout)
    result = pe(X)
    print(result)


def test_TransformerDecoder():
    bz = 2
    sz = 10
    dm = 16
    voca_size = 3000
    num_hidden = 60
    num_head = 4
    dropout = 0.1
    num_layer = 3
    shape = (bz, sz)
    X = torch.randint(0, voca_size, shape)
    valid_lens = [7, 9]
    valid_lens = torch.tensor(valid_lens)
    encoder = TransformerEncoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    result = encoder(X, valid_lens)
    print(X)
    print(result.shape)


key_values = [None] * 5
for j in range(3):
    for i in range(5):
        print(i)
        X = torch.ones((1, 1, 4)) + j
        if key_values[i] is None:
            print("none")
            key_values[i] = X
        else:
            print("not none")
            key_values[i] = torch.cat((key_values[i], X), dim=1)
        print(key_values[i])
