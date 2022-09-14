import cv2, os, pickle, numpy as np
from tqdm import tqdm
from .common import size, arr_size, ansmap, lenA

def read(name, terdir, frag):
    pkl = f'out/src/{name}.pkl'
    video = f'{terdir}/{name}.mp4'
    edited = f'out/edited/{name}.mp4'

    cap = cv2.VideoCapture(video)
    arr = np.array([])

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_ma = int((h-w)/2) if w < h else 0
    w_ma = int((w-h)/2) if w > h else 0

    if frag or not os.path.isfile(pkl):
        print('dumping '+video)
        fmt = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(edited,fmt,cap.get(cv2.CAP_PROP_FPS),(size,size),0)
        with open(pkl, 'wb') as f:
            for _ in tqdm(range(frame_count)):
                _, frame = cap.read()
                frame = cv2.resize(frame[h_ma:h-h_ma, w_ma:w-w_ma], dsize=(size, size))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = np.array([cv2.split(gray)])
                arr = gray if arr.size == 0 else np.append(arr, gray, axis=0)
            print('editting '+video)
            all_m = arr.mean(0)
            for i in tqdm(range(frame_count)):
                ab = np.abs(arr[i]-all_m)
                arr[i] = np.where((all_m/4<ab) & (ab<all_m), 255, 0)
            bin_sort = np.argsort(np.abs(arr.mean(0)-255/2), axis=None)
            tar0, tar1, tar2 = np.unravel_index(bin_sort, (1,size,size))
            for i in tqdm(range(frame_count)):
                cond = 500
                for n, (d0,d1,d2) in enumerate(zip(tar0,tar1,tar2)):
                    if arr[i,d0,d1,d2] == 0:
                        cond += 1
                    elif n == cond:
                        target = (tar0[:cond],tar1[:cond],tar2[:cond])
                        break
                arr[i], arr[i][target] = 0, arr[i][target]
                writer.write(cv2.merge(arr[i]))
            pickle.dump(arr, f)

    print('loading '+pkl)
    with open(pkl, 'rb') as f:
        return pickle.load(f)

def all_read(name, force=False):
    if force if force else input('update data [y/n]: ') == 'y':
        data, teachs, plot = np.array([]), np.array([]), np.array([])
        other = [0 for _ in range(lenA-1)]+[1]
        for s in os.listdir(name):
            s = s.replace('.mp4', '')
            arr = read(s, name, force)
            data = arr if data.size == 0 else np.append(data, arr, axis=0)
            s = s.split('_')[-1]
            teach = np.array([ansmap.get(s, other) for _ in range(len(arr)-arr_size)])
            teachs = teach if teachs.size == 0 else np.append(teachs, teach, axis=0)
            plot = np.append(plot, len(data)-arr_size)
        with open(f'out/model/{name}_read.pkl', 'wb') as f:
            print('dumping '+f'out/model/{name}_read.pkl')
            pickle.dump((data, teachs, plot), f)
        return data, teachs, plot
    else:
        with open(f'out/model/{name}_read.pkl', 'rb') as f:
            print('loading '+f'out/model/{name}_read.pkl')
            return pickle.load(f)

def test_read():
    if input('read video data [y/n]: ') == 'y':
        with open('out/model/video_read.pkl', 'rb') as f:
            print('loading '+'out/model/video_read.pkl')
            return pickle.load(f)
    else:
        return all_read('test')