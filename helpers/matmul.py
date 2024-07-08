import sys
import time
import argparse

import torch
import numpy as np


def export_tensor(t, fp):
    t = t.numpy()
    header = [0, t.itemsize, t.size, t.ndim] + list(t.shape)
    header = [len(header)] + header
    header = np.array(header).astype('int32')
    with open(fp, 'wb') as f:
        header.tofile(f)
        t.tofile(f)
    print(header)
    print(t.ravel()[:8])
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('a_shape')
    parser.add_argument('b_shape')
    args = parser.parse_args()
    a_shape = tuple(map(int, args.a_shape.split('x')))
    b_shape = tuple(map(int, args.b_shape.split('x')))
    
    a = torch.rand(*a_shape).float()
    export_tensor(a, 'a.bin')
    b = torch.rand(*b_shape).float()
    export_tensor(b, 'b.bin')
    torch.matmul(a, b)  # warmup
    st = time.perf_counter()
    c = torch.matmul(a, b)
    rt = time.perf_counter() - st
    export_tensor(c, 'c.bin')
    print(f'shape={c.shape}\ttime={rt}')


if __name__ == '__main__':
    main()
