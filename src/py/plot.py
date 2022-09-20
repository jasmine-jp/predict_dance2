import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute

    def saveimg(self, m, ans, idx):
        conv, enc, rnn, sound = map(lambda l: l[0].detach().clone(), (m.c,m.e,m.r,m.s))
        fig = plt.figure(figsize=(24, 4), tight_layout=True)
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)
        ax1.set_title('conv')
        ax2.set_title('sound')
        ax3.set_title('encode')
        ax4.set_title('rnn')
        ax1.plot(list(map(float, conv.mean(1))))
        ax2.plot(list(map(float, sound.mean(1))))
        ax3.plot(list(map(float, enc.transpose(0,1).mean(1))))
        for r in rnn:
            ax4.plot(list(map(float, r)))
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        plt.close(fig)
        fig.savefig(f'out/img/{s}/estimate_{idx}')