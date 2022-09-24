import cv2, os, pickle, librosa, ffmpeg as fp, numpy as np
from tqdm import tqdm
initial_rang = 500
sound_size = 2**11

class Read:
    def __init__(self, dirname, size, force):
        self.dirname = dirname
        self.size = (size, size)
        self.force = force

    def read(self, filename):
        self.name = filename
        video = f'{self.dirname}/{filename}.mp4'
        vpkl = f'out/src/video/{filename}.pkl'
        sound = f'out/edited/sound/{filename}.mp3'
        spkl = f'out/src/sound/{filename}.pkl'

        if self.force or not os.path.isfile(vpkl):
            self.cap = cv2.VideoCapture(video)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            with open(vpkl, 'wb') as f:
                raw, mask = self.RGB2GRAY()
                mask = self.GRAY2BIN(mask)
                raw = self.BIN2RAW(raw, mask)
                pickle.dump(raw, f)

        if self.force or not os.path.isfile(spkl):
            print('making mp3 from', video)
            fp.run(fp.output(fp.input(video),sound),quiet=True,overwrite_output=True)
            y, _ = librosa.load(sound, sr=61400)
            long = sound_size*self.frame_count
            y = np.append(y,[0 for _ in range(long-len(y))]) if long-len(y)>0 else y[:long]
            y = y.reshape((self.frame_count, -1))
            with open(spkl, 'wb') as f:
                pickle.dump(y, f)

        print('loading', vpkl, 'and', spkl)
        with open(vpkl, 'rb') as v, open(spkl, 'rb') as s:
            return pickle.load(v), pickle.load(s)

    def RGB2GRAY(self):
        print('graying '+ f'{self.dirname}/{self.name}.mp4')
        raw, mask = np.array([]), np.array([])

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        h_ma = int((h-w)/2) if w < h else 0
        w_ma = int((w-h)/2) if w > h else 0

        for _ in tqdm(range(self.frame_count)):
            _, frame = self.cap.read()
            frame = cv2.resize(frame[h_ma:h-h_ma, w_ma:w-w_ma], self.size)
            rgb = np.array([cv2.split(frame)])
            raw = rgb if raw.size == 0 else np.append(raw, rgb, axis=0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = np.array([cv2.split(gray)])
            mask = gray if mask.size == 0 else np.append(mask, gray, axis=0)
        return raw, mask

    def GRAY2BIN(self, mask, rang=initial_rang):
        print('masking '+ f'{self.dirname}/{self.name}.mp4')

        mask = np.where(mask<=mask.mean((1,2,3)).reshape((-1,1,1,1)),255,0).astype(np.uint8)
        bin_sort = np.argsort(np.abs(mask.mean(0)-255/4), axis=None)
        tar0, tar1, tar2 = np.unravel_index(bin_sort, mask.shape[1:])

        for i in tqdm(range(self.frame_count)):
            for n, (d0,d1,d2) in enumerate(zip(tar0,tar1,tar2)):
                if mask[i,d0,d1,d2] == 0:
                    rang += 1
                if n == rang or n == np.prod(self.size)-1:
                    target = tuple(map(lambda e: e[:n],(tar0,tar1,tar2)))
                    rang = initial_rang
                    break
            mask[i], mask[i][target] = 0, mask[i][target]
        return mask
    
    def BIN2RAW(self, raw, mask):
        print('editting '+ f'{self.dirname}/{self.name}.mp4')
        arr = np.array([])

        edited = f'out/edited/video/{self.name}.mp4'
        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(edited, fmt, self.fps, self.size)

        for r, m in zip(raw, mask):
            r = np.ma.masked_where(list(m==0)*3, r).filled(0)
            arr=r[np.newaxis] if arr.size==0 else np.append(arr,r[np.newaxis],axis=0)
            writer.write(cv2.merge(r))
        return arr