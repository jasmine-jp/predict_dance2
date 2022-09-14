import torch, torchinfo
from src.py.network import NeuralNetwork
from src.py.study import Study
from src.py.common import all_read, test_read, arr_size, size
from src.py.plot import plot
# print('Input Size:', 10, arr_size, 1, size, size)
# torchinfo.summary(NeuralNetwork(), (10, arr_size, 1, size, size))

model = torch.load('out/model/model_weights.pth')
study = Study(model, test_read(), 750, plot(False))

study.test()