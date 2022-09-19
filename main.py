import os, shutil, torch
from src.py.study import Study
from src.py.network import NeuralNetwork
from src.py.common import all_read
from src.py.plot import plot

model = NeuralNetwork()
study = Study(model, all_read('video'), 5000, plot(True))
loss = 1

epochs = 20
shutil.rmtree('out/img', ignore_errors=True)
os.mkdir('out/img')
os.mkdir(f'out/img/test')
for idx in range(1, epochs+1):
    print(f'Epoch {idx}\n-------------------------------')
    os.mkdir(f'out/img/epoch_{idx}')
    study.p.epoch = idx
    study.train()
    study.test()
    if loss > study.test_loss:
        print('Saving PyTorch Model State')
        torch.save(model, 'out/model/model_weights.pth')
        loss = study.test_loss
print(f'final loss: {loss}')