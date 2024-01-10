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


def test_TransformerEncoder():
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


def test_DecoderLayer_1():
    bz = 2
    sz = 10
    dm = 16
    num_hidden = 60
    num_head = 4
    dropout = 0.1
    num_layer = 2
    X = torch.randn((bz, sz, dm))
    Y = torch.randn((bz, sz, dm))
    enc_valid = torch.tensor([6, 9])
    dec_valid = torch.tensor([7, 8])
    el = EncodeLayer(dm, num_hidden, num_head, dropout)
    enc_output = el(X, enc_valid)
    dls = [
        DecodeLayer(idx, dm, num_hidden, num_head, dropout) for idx in range(num_layer)
    ]
    KV = [None] * num_layer
    dec_input = (Y, KV, dec_valid)
    enc_info = (enc_output, enc_valid)
    for dl in dls:
        dec_input, enc_info = dl(dec_input, enc_info)
        print(dec_input[0].shape)
        print(KV[0].shape)


def test_DecoderLayer_2():
    bz = 1
    sz = 10
    dm = 16
    num_hidden = 60
    num_head = 4
    dropout = 0.1
    num_layer = 2
    epchos = 3
    X = torch.randn((bz, sz, dm))
    Y = torch.randn((bz, 1, dm))
    enc_valid = torch.tensor([6])
    dec_valid = None
    el = EncodeLayer(dm, num_hidden, num_head, dropout)
    enc_output = el(X, enc_valid)
    dls = [
        DecodeLayer(idx, dm, num_hidden, num_head, dropout) for idx in range(num_layer)
    ]
    KV = [None] * num_layer
    dec_input = (Y, KV, dec_valid)
    enc_info = (enc_output, enc_valid)
    savd_Y = []
    for j in range(epchos):
        for dl in dls:
            savd_Y.append(dec_input[0])
            dec_input, enc_info = dl(dec_input, enc_info)
        print(Y.shape)
        print(dec_input[1][0].shape)
    for j in range(epchos):
        for i in range(num_layer):
            print(savd_Y[i + j * num_layer] - dec_input[1][i][0, j, :] < 1e-6)


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
    Y = torch.randint(0, voca_size, shape)
    enc_valid = torch.tensor([7, 9])
    dec_valid = torch.tensor([8, 6])
    encoder = TransformerEncoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    decoder = TransformerDecoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    enc_output = encoder(X, enc_valid)
    result = decoder(Y, dec_valid, enc_output, enc_valid)
    print(X.shape)
    print(Y.shape)
    print(enc_output.shape)
    print(result.shape)


def test_TransformerDecoder2():
    bz = 1
    sz = 10
    dm = 16
    voca_size = 3000
    num_hidden = 60
    num_head = 4
    dropout = 0.1
    num_layer = 3
    shape = (bz, sz)
    X = torch.randint(0, voca_size, shape)
    Y = torch.randint(0, voca_size, (1, 1))
    enc_valid = torch.tensor([8])
    dec_valid = None
    encoder = TransformerEncoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    decoder = TransformerDecoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    enc_output = encoder(X, enc_valid)
    decoder.eval()
    for i in range(3):
        Y_hat = decoder(Y, dec_valid, enc_output, enc_valid)
        Y = torch.argmax(Y_hat, dim=-1)
        print(Y)
        print(decoder.key_value[0].shape)
        print("*******************************")


def test_Transformer():
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
    Y = torch.randint(0, voca_size, (bz, sz))
    enc_valid = torch.tensor([8, 7])
    dec_valid = torch.tensor([6, 9])
    encoder = TransformerEncoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    decoder = TransformerDecoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    trf = Transformer(encoder, decoder)
    result = trf(X, Y, enc_valid, dec_valid)
    print(result.shape)


def test_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    files = [
        f"datasets\wikitext-103-raw\wiki.{split}.raw"
        for split in ["test", "train", "valid"]
    ]
    tokenizer.train(files, trainer)
    tokenizer.save("datasets/tokenizer-wiki.json")
    # tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")


def test_read_data():
    filepath = "datasets/translation2019zh_valid.json"
    ch_data, en_data = read_data(filepath)
    print(ch_data[0:10])
    print(en_data[0:10])


def test_my_tokenizer():
    # filepath = "datasets/translation2019zh_valid.json"
    # ch_data, en_data = read_data(filepath)
    # tokenizer = my_tokenizer(ch_data)
    tokenizer = Tokenizer.from_file("ch.json")

    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=10)
    num_seq = tokenizer.encode("他开始了工作！")
    mask = num_seq.attention_mask
    print(num_seq.tokens)
    print(num_seq.ids)
    print(mask)
    print(torch.sum(torch.tensor(mask)).item())
    print(num_seq)

    ch_seq = tokenizer.decode(num_seq.ids)
    print(ch_seq)


def custom_dataloader(bz, shuffle=True, num_workers=0):
    filepath = "datasets\\translation2019zh_valid.json"
    chinese_data, english_data = read_data(filepath)
    dataset = CustomDataset(chinese_data, english_data)
    dataloader = DataLoader(dataset, bz, shuffle=shuffle, num_workers=num_workers)
    data, label = next(iter(dataloader))
    print(data[0])
    print(label[0])


def test_train():
    bz = 2
    sz = 64
    dm = 64
    voca_size = 30000
    num_hidden = 60
    num_head = 4
    dropout = 0.1
    num_layer = 3
    encoder = TransformerEncoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    decoder = TransformerDecoder(
        voca_size, dm, num_hidden, num_head, dropout, num_layer
    )
    net = Transformer(encoder, decoder)

    # 处理数据，得到词表。这里应该分为中文词表和英文词表
    feature, label, enc_tokenizer, dec_tokenizer = load_tokenizer()

    # 将数据分为训练集和测试集
    data_size = len(feature)
    train_data = CustomDataset(
        feature[0 : int(data_size * 0.8)], label[0 : int(data_size * 0.8)]
    )
    # 得到DataLoader
    train_dataloader = DataLoader(train_data, bz, shuffle=True)
    # 初始化模型
    loss = nn.CrossEntropyLoss(reduction="none")

    # 进行训练
    for _ in range(3):
        for iter in train_dataloader:
            X, Y = iter[0], iter[1]
            X = enc_tokenizer.encode_batch(X)
            enc_valid = torch.tensor([sum(x.attention_mask) for x in X])
            Y = dec_tokenizer.encode_batch(Y)
            Y_mask = torch.tensor([y.attention_mask for y in Y])
            dec_valid = torch.tensor([sum(y) for y in Y_mask])
            X = torch.tensor([x.ids for x in X])
            Y = torch.tensor([y.ids for y in Y])
            print(X.shape)
            print(Y.shape)
            print(enc_valid)
            print(dec_valid)
            dec_output = net(X, Y, enc_valid, dec_valid)
            unweighted_loss = loss(dec_output.permute(0, 2, 1), Y)
            print(unweighted_loss.shape)
            weighted_loss = (unweighted_loss * Y_mask).mean(dim=1)
            weighted_loss.sum().backward()
            print(weighted_loss)


def test_load_tokenizer():
    feature, label, enc_tok, dec_tok = load_tokenizer()
    print(feature[0])
    print(label[0])
    print(enc_tok.encode("我使大家高兴！").ids)
    print(dec_tok.encode("I make everybody happy!").ids)


test_train()
