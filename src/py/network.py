import torch
from torch import nn
from .common import *
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.cl = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, sec_d, sec_size, sec_size),
            nn.BatchNorm2d(sec_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.BatchNorm2d(thr_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten(),
            nn.Dropout(0.2)
        ) for _ in range(batch)])

        e = nn.TransformerEncoderLayer(out,8,batch_first=True)
        self.encoder = nn.TransformerEncoder(e, 2)

        self.rnn = nn.ModuleList([nn.GRU(out,hidden,batch_first=True) for _ in ians])
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
        self.c = torch.stack(list(map(lambda c,e:c(e),self.cl,x)))
        self.e = self.encoder(self.c)
        self.r = torch.stack(list(map(self.arrange, self.rnn, ians))).transpose(0,1)
        return self.stack(self.r)

    def arrange(self, r, i):
        o, hn = r(self.c, self.hn[i])
        self.hn[i].data = hn if self.training else self.hn[i]
        return o[:, :, -1]