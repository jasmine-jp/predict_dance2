import torch
from torch import nn
from .common import *
reng = 3700
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.sound = torch.zeros((batch, arr_size, sound_size))
        self.pos = torch.zeros((batch, arr_size), dtype=int)

        self.video_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, sec_d, sec_size, sec_size),
            nn.BatchNorm2d(sec_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.BatchNorm2d(thr_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten()
        ) for _ in range(batch)])

        self.sound_conv = nn.Sequential(
            nn.Conv1d(arr_size, 45, 2, 2),
            nn.BatchNorm1d(45),
            nn.Tanh(),
            nn.AvgPool1d(2),
            nn.Conv1d(45, arr_size, 2, 2),
            nn.BatchNorm1d(arr_size),
            nn.Tanh(),
            nn.AvgPool1d(2)
        )

        self.embedding = nn.Embedding(reng, out)
        e = nn.TransformerEncoderLayer(out, 8, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(e, 6)

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
        self.c = torch.stack(list(map(lambda c, e: c(e), self.video_conv, x)))
        self.e = self.encoder(self.embedding(self.pos) + self.c)
        self.s = self.sound_conv(self.sound)
        self.r = torch.stack(list(map(self.arrange, self.rnn, ians))).transpose(0,1)
        return self.stack(self.r)

    def arrange(self, r, i):
        o, hn = r(self.e + self.s, self.hn[i])
        self.hn[i].data = hn if self.training else self.hn[i]
        return o[:, :, -1]

    def setstate(self, sound, pos):
        self.sound = sound
        self.pos = pos