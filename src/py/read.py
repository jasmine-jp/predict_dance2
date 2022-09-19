import cv2, os, pickle, librosa, ffmpeg as fp, numpy as np
from tqdm import tqdm
initial_rang = 500

class Read:
    def __init__(self, dirname, size, force):
        self.dirname = dirname
        self.size = (size, size)
        self.force = force

    def read(self, filename):
        arr = np.array([])
        video = f'{self.dirname}/{filename}.mp4'
        vpkl = f'out/src/video/{filename}.pkl'
        sound = f'out/edited/sound/{filename}.mp3'
        spkl = f'out/src/sound/{filename}.pkl'

        if self.force or not os.path.isfile(vpkl):
            self.cap = cv2.VideoCapture(video)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            with open(vpkl, 'wb') as f:
                arr = self.RGB2GRAY(filename, arr)
                arr = self.GRAY2BIN(filename, arr)
                pickle.dump(arr, f)

        if self.force or not os.path.isfile(spkl):
            print('making mp3 from', video)
            fp.run(fp.output(fp.input(video),sound),quiet=True,overwrite_output=True)
            y, _ = librosa.load(sound)
            rest = 735*self.frame_count-len(y)
            y = np.append(y,[0 for _ in range(rest)]) if rest>0 else y[:735*self.frame_count]
            y = y.reshape((self.frame_count, -1))
            with open(spkl, 'wb') as f:
                pickle.dump(y, f)

        print('loading', vpkl, 'and', spkl)
        with open(vpkl, 'rb') as v, open(spkl, 'rb') as s:
            return pickle.load(v), pickle.load(s)

    def RGB2GRAY(self, name, arr):
        print('be gray', f'{self.dirname}/{name}.mp4')

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        h_ma = int((h-w)/2) if w < h else 0
        w_ma = int((w-h)/2) if w > h else 0

        for _ in tqdm(range(self.frame_count)):
            _, frame = self.cap.read()
            frame = cv2.resize(frame[h_ma:h-h_ma, w_ma:w-w_ma], self.size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = np.array([cv2.split(gray)])
            arr = gray if arr.size == 0 else np.append(arr, gray, axis=0)
        return arr

    def GRAY2BIN(self, name, arr, rang=initial_rang):
        print('editting', f'{self.dirname}/{name}.mp4')
        edited = f'out/edited/video/{name}.mp4'

        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(edited, fmt, self.fps, self.size, 0)

        arr = np.where(arr<=arr.mean((1,2,3)).reshape((-1,1,1,1)),255,0).astype(np.uint8)
        bin_sort = np.argsort(np.abs(arr.mean(0)-255/4), axis=None)
        tar0, tar1, tar2 = np.unravel_index(bin_sort, arr.shape[1:])

        for i in tqdm(range(self.frame_count)):
            for n, (d0,d1,d2) in enumerate(zip(tar0,tar1,tar2)):
                if arr[i,d0,d1,d2] == 0:
                    rang += 1
                if n == rang or n == np.prod(self.size)-1:
                    target = tuple(map(lambda e: e[:n],(tar0,tar1,tar2)))
                    rang = initial_rang
                    break
            arr[i], arr[i][target] = 0, arr[i][target]
            writer.write(cv2.merge(arr[i]))
        return arr