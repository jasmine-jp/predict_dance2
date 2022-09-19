import torch
from torch import nn
from .common import *
ians, emsize, reng = range(lenA), int(out/2), 3700
zeros = torch.zeros((1, batch, hidden))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = torch.zeros((batch, arr_size), dtype=int)

        self.convL = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, sec_d, sec_size, sec_size),
            nn.BatchNorm2d(sec_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.BatchNorm2d(thr_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten()
        ) for _ in range(batch)])

        self.embedding = nn.Embedding(reng, emsize)
        t = out + emsize

        e = nn.TransformerEncoderLayer(t, 8, batch_first=True)
        self.encoder = nn.TransformerEncoder(e, 3)

        self.rnn = nn.ModuleList([nn.GRU(t,hidden,batch_first=True) for _ in ians])
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lenA*arr_size, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, lenA),
            nn.Softmax(1)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(lambda conv, e: conv(e), self.convL, x)))
        self.e = self.encoder(torch.cat((self.embedding(self.pos), self.c), dim=2))
        self.r = torch.stack(list(map(self.arrange, self.rnn, ians))).transpose(0,1)
        return self.stack(self.r)

    def arrange(self, r, i):
        o, hn = r(self.e, self.hn[i])
        self.hn[i].data = hn if self.training else self.hn[i]
        return o[:, :, -1]

    def setstate(self, pos):
        self.pos = torch.tensor(pos, dtype=int)