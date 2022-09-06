from sympy import factorint

ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0]
}
lenA, out, el = len(ansmap)+1, 16, 4
size, arr_size, pool = 64, 32, 2
diff = size/(pool**2)/(el**0.5)
sec_d, thr_d = int(out/el/2), int(out/el)
sec_size, thr_size = int(diff/2), 2
batch, hidden = 10, arr_size