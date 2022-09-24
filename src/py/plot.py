import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute

    def saveimg(self, m, ans, idx):
        conv, enc = map(lambda l: l[0].detach().clone(), (m.c,m.e))
        fig = plt.figure(figsize=(12, 4), tight_layout=True)
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title('conv + pixel_embedding')
        ax2.set_title('encode')
        ax1.plot(list(map(float, conv.mean(1))))
        ax2.plot(list(map(float, enc.transpose(0,1).mean(1))))
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        plt.close(fig)
        fig.savefig(f'out/img/{s}/estimate_{idx}')