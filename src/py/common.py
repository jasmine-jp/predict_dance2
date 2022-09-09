ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0]
}
lenA, out, el = len(ansmap)+1, 32, 4
size, arr_size, pool = 64, 90, 2
diff, last = size/(pool**2)/(el**0.5), 4
sec_d, thr_d = int(out/el/2), int(out/el)
sec_size, thr_size = int(diff/last), last
batch, hidden = 10, arr_size