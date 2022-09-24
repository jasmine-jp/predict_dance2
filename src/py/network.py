import torch
from torch import nn
from .common import *
pixel_idx = torch.arange(0,out).repeat(batch*arr_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.convL = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, sec_d, sec_size, sec_size),
            nn.BatchNorm2d(sec_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.BatchNorm2d(thr_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten(),
            nn.Dropout(0.5)
        ) for _ in range(batch)])

        self.pixel_embedding = nn.Sequential(
            nn.Embedding(out, 1),
            nn.Dropout(0.5)
        )

        e = nn.TransformerEncoderLayer(out, 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(e, 2)

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out*arr_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(8, lenA),
            nn.Softmax(1)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(lambda conv, e: conv(e), self.convL, x)))
        embedding = self.pixel_embedding(pixel_idx).reshape(self.c.shape)
        self.c = embedding + self.c
        self.e = self.encoder(self.c)
        return self.stack(self.e)