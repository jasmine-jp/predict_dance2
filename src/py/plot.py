import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute

    def saveimg(self, c, r, ans, idx):
        conv, rnn = c[0].detach().clone(), r[0].detach().clone()
        fig = plt.figure(figsize=(12.8, 4.8))
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title('conv')
        ax2.set_title('rnn')
        for c in conv:
            ax1.plot(list(map(float, c)))
        for r in rnn:
            ax2.plot(list(map(float, r)))
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        plt.close(fig)
        fig.savefig(f'out/img/{s}/estimate_{idx}')