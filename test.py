from chinese2english import *
import numpy as np
from d2l import torch as d2l


def test_DotProductAttention():
    bz = 2
    sz = 10
    dm = 16
    dropout = 0.01
    q = torch.rand((bz, sz, dm))
    k = torch.rand((bz, sz, dm))
    v = torch.rand((bz, sz, dm))
    # valid = [torch.randint(0, sz, (1, 1)).item() for x in range(bz)]
    # print(valid)
    # valid = torch.repeat_interleave(torch.tensor(valid), sz).reshape(bz, -1)
    valid = torch.arange(1, sz + 1).expand(bz, sz)
    dpa = DotProductAttention(dropout)
    result = dpa(q, k, v, valid)
    print(q)
    print(k)
    print(v)
    print(result)
    print(result.shape)


# test_DotProductAttention()


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


def test_expand():
    # q = torch.rand((2, 3, 4))
    # valid = torch.arange(0, 3)
    # print(valid)
    # print(valid.shape)
    # # 等价于先unsqueeze再repeat
    # valid = valid.expand(2, 3).unsqueeze(2)
    # print(valid.shape)
    # # valid = valid.repeat(2, 1)
    # # print(valid.shape)
    # print(valid)
    # mask = torch.arange(0, q.shape[1]).expand((2, q.shape[1], q.shape[1]))
    # print(mask)
    # mask = mask > valid
    # print(mask)\
    bz = 4
    sz = 10
    num_head = 4
    # valid = torch.arange(1, sz + 1).expand(bz, sz)
    # valid = [torch.randint(0, sz, (1, 1)).item() for x in range(bz)]
    # print(valid)
    # valid = torch.repeat_interleave(torch.tensor(valid), sz).reshape(bz, -1)
    # print(valid)
    # valid = valid.unsqueeze(1).expand(bz, num_head, sz).reshape(-1, sz)
    # print(valid)
    a = torch.tensor([3, 4, 5])
    b = a.expand(2, 3)
    print(b)
    c = b.unsqueeze(2).expand(2, 3, 3)
    print(c)


def test_custom_data():
    feature_vocab_size = 10000
    label_vocab_size = 3000
    feature_seq_size = 128
    label_seq_size = 128
    X, X_valid, X_mask, Y, Y_valid, Y_mask, feature, label = custom_data(
        feature_vocab_size, label_vocab_size, feature_seq_size, label_seq_size
    )
    for i in range(len(X)):
        print(X[i])
        print(X_valid[i])
        print(X_mask[i])
        print(Y[i])
        print(Y_valid[i])
        print(Y_mask[i])
        print(feature[i])
        print(label[i])


def test_run():
    epochs = 1
    bz = 128
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_path = "datasets/train_XY.pt"
    X, X_valid, X_mask, Y, Y_valid, Y_mask = torch.load(train_path)
    dataset = CustomDataset(X, X_valid, X_mask, Y, Y_valid, Y_mask)
    dataloader = DataLoader(dataset, bz, shuffle=False)
    vz1 = 10000
    vz2 = 3000
    sz = 128
    dm = 128
    num_heads = 8
    num_hidden = 512
    num_layer = 1
    dropout = 0.001
    encoder = TransformerEncoder(vz1, dm, num_hidden, num_heads, dropout, num_layer)
    decoder = TransformerDecoder(vz2, dm, num_hidden, num_heads, dropout, num_layer)
    model = Transformer(encoder, decoder)
    c2e = Chinese2English()
    c2e.train(model, dataloader, epochs, lr, device)
