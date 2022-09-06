import torch
from torch import nn
from .common import *
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = 'train'

        self.conv2d = nn.Sequential(
            nn.Conv2d(lenA, sec_d, sec_size, sec_size),
            nn.BatchNorm2d(sec_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.BatchNorm2d(thr_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten()
        )

        enc_layer = nn.TransformerEncoderLayer(out,2,batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)

        self.rnn = nn.ModuleList([nn.GRU(out,hidden,batch_first=True) for _ in ians])
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        dec_layer = nn.TransformerDecoderLayer(arr_size,2,batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, 1)

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lenA*arr_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, lenA)
        )

    def forward(self, x):
        c = torch.stack(list(map(self.conv2d, x)))
        self.e = self.encoder(c)
        r = torch.stack(list(map(self.arrange, self.rnn, ians))).transpose(0,1)
        d = self.decoder(r, self.e.transpose(1,2))
        return self.stack(d)

    def arrange(self, r, i):
        o, hn = r(self.e, self.hn[i])
        self.hn[i].data = hn if self.s == 'train' else self.hn[i]
        return o[:, :, -1]

    def setstate(self, s):
        self.s = s