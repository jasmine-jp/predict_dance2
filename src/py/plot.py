import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute

    def saveimg(self, c, ans, idx):
        conv = c[0].detach().clone()
        fig = plt.figure(figsize=(6.8, 4.8))
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot()
        ax1.set_title('conv')
        ax1.plot(list(map(float, conv.mean(1))))
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        plt.close(fig)
        fig.savefig(f'out/img/{s}/estimate_{idx}')