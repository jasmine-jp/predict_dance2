import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute

    def saveimg(self, c, e, ans, idx):
        conv, enc = c[0].detach().clone(), e[0].detach().clone()
        fig = plt.figure(figsize=(12.8, 4.8))
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title('conv')
        ax2.set_title('encode')
        ax1.plot(list(map(float, conv.mean(1))))
        ax2.plot(list(map(float, enc.mean(1))))
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        plt.close(fig)
        fig.savefig(f'out/img/{s}/estimate_{idx}')