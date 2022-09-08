import torch, numpy as np
from tqdm import tqdm
from .common import arr_size, lenA, batch

class Study:
    def __init__(self, model, read, diff, p):
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.RAdam(model.parameters())
        self.model, self.p = model, p
        self.data, self.teach, self.plot = read
        self.diff = np.array([len(self.teach)-diff, diff])/batch

    def train(self):
        self.p.test, d = False, int(self.diff[0])
        print('train')

        for i in tqdm(range(1, d+1)):
            train, teach = self.create_randrange()
            self.model.setstate('train')
            pred = self.model(train)
            loss = self.loss_fn(pred, teach)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i % 300 == 0 or i == 1) and self.p.execute:
                self.p.saveimg(self.model.c, self.model.r, teach, i)

    def test(self):
        self.p.test, d = True, int(self.diff[1])
        self.test_loss, co = 0, 0
        psum, ans = [torch.zeros(lenA) for _ in range(2)]
        print('test')

        with torch.no_grad():
            for i in tqdm(range(1, d+1)):
                train, teach = self.create_randrange()
                self.model.setstate('test')
                pred = self.model(train)
                self.test_loss += self.loss_fn(pred, teach).item()

                for m, t in zip(pred.argmax(dim=1), teach):
                    co, psum[m], ans = co+t[m], psum[m]+1, ans+t

                if (i % 100 == 0 or i == 1) and self.p.execute:
                    self.p.saveimg(self.model.c, self.model.r, teach, i)

            self.test_loss, co = self.test_loss/d, co/d/batch
            print(f'Accuracy: {(100*co):>0.1f}%, Avg loss: {self.test_loss:>8f}')
            print(f'Sum: {list(map(int, psum))}, Ans: {list(map(int, ans))}')

    def create_randrange(self):
        r = np.random.randint(0, len(self.data), batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.plot-e)), r)))
        trainE = np.array(list(map(lambda e, i: e-arr_size if 0<e-self.plot[i]<arr_size else e, r, idx)))
        trainNum = np.array(list(map(lambda e: np.arange(e, e+arr_size), trainE)))
        teachNum = np.array(list(map(lambda e, i: e-(i if e < self.plot[i] else i+1)*arr_size, trainE, idx)))
        return torch.Tensor(self.data[trainNum]), torch.Tensor(self.teach[teachNum])
