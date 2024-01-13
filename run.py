from chinese2english import *

epochs = 150
bz = 800
lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
train_path = "datasets/train_XY.pt"
X, X_valid, X_mask, Y, Y_valid, Y_mask = torch.load(train_path)
dataset = CustomDataset(X, X_valid, X_mask, Y, Y_valid, Y_mask)
dataloader = DataLoader(dataset, bz, shuffle=True)
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
