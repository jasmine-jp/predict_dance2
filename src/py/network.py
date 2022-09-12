import torch
from torch import nn
from .common import *
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(lenA, sec_d, sec_size, sec_size),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten()
        )

        self.el = nn.TransformerEncoderLayer(out,1,batch_first=True)

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out*arr_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, lenA),
            nn.Softmax(1)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x)))
        self.e = self.el(self.c)
        return self.stack(self.e)