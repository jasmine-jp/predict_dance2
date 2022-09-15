import torch
from torch import nn
from .common import *

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
            nn.Flatten()
        ) for _ in range(batch)])

        e = nn.TransformerEncoderLayer(out,1,batch_first=True)
        self.encoder = nn.TransformerEncoder(e, 2)

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out*arr_size, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, lenA),
            nn.Softmax(1)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(lambda c,e:c(e),self.cl,x)))
        self.e = self.encoder(self.c)
        return self.stack(self.e)