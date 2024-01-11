from chinese2english import *
import numpy as np
from d2l import torch as d2l


def test_DotProductAttention():
    bz = 2
    sz = 3
    dm = 4
    dropout = 0.01
    q = torch.rand((bz, sz, dm))
    k = torch.rand((bz, sz, dm))
    v = torch.rand((bz, sz, dm))
    mask = []
    for i in range(bz):
        n = int(torch.randint(1, sz, (1, 1)).item())
        tmp = []
        for j in range(sz):
            if j < n:
                tmp.append(1)
            else:
                tmp.append(0)
        mask.append(tmp)
    mask = torch.tensor(mask)
    print(mask)
    dpa = DotProductAttention(dropout)
    result = dpa(q, k, v, mask)
    print(q)
    print(k)
    print(v)
    print(result)
    print(result.shape)


def test_MultiHeadAttention():
    bz = 2
    sz = 3
    dm = 4
    dropout = 0.01
    num_head = 2
    mask = []
    for i in range(bz):
        n = int(torch.randint(1, sz, (1, 1)).item())
        tmp = []
        for j in range(sz):
            if j < n:
                tmp.append(1)
            else:
                tmp.append(0)
        mask.append(tmp)
    mask = torch.tensor(mask)
    print(mask)
    dpa = MultiHeadAttention(dm, dm, dm, num_head, dropout)
    dpa.train()
    optimizer = torch.optim.Adam(dpa.parameters(), lr=0.001)
    for i in range(3):
        print(i)
        q = torch.rand((bz, sz, dm))
        k = torch.rand((bz, sz, dm))
        v = torch.rand((bz, sz, dm))
        result = dpa(q, k, v, mask)
        l = (result * 2).sum().sum().sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


def test_mask():
    num_head = 4
    mask = torch.randint(0, 10, (2, 3))
    print(mask)
    new_mask = mask.unsqueeze(1).expand(-1, num_head, -1).reshape(-1, mask.shape[1])
    print(new_mask)
    new2_mask = new_mask.unsqueeze(1).expand(-1, mask.shape[1], -1)
    print(new2_mask)


def test_TransformerEncoder():
    vz = 3000
    bz = 2
    sz = 3
    dm = 16
    dropout = 0.001
    num_head = 4
    num_layer = 2
    num_hidden = 64

    tfenc = TransformerEncoder(vz, dm, num_hidden, num_head, dropout, num_layer)
    tfenc.train()
    optimizer = torch.optim.Adam(tfenc.parameters(), lr=0.001)
    for iter in range(3):
        print(iter)
        X = torch.randint(0, vz, (bz, sz))
        mask = []
        for i in range(bz):
            n = int(torch.randint(1, sz, (1, 1)).item())
            tmp = []
            for j in range(sz):
                if j < n:
                    tmp.append(1)
                else:
                    tmp.append(0)
            mask.append(tmp)
        mask = torch.tensor(mask)
        result = tfenc(X, mask)
        l = (result * 2).sum().sum().sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


test_TransformerEncoder()
