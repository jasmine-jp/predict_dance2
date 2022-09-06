import torch
from torch import nn
from .common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

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

        enc_layer = nn.TransformerEncoderLayer(out, 2)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        dec_layer = nn.TransformerDecoderLayer(out, 2)
        self.decoder = nn.TransformerDecoder(dec_layer, 1)

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(arr_size*out, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, lenA)
        )

    def forward(self, x):
        c = torch.stack(list(map(self.conv2d, x)))
        e = self.encoder(c)
        d = self.decoder(e, c)
        return self.stack(d)