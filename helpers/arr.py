import sys

import numpy as np

DTYPES = {'float32': 0}

def export_array(a, fp):
    header = [DTYPES[str(a.dtype)], a.itemsize, a.size, a.ndim] + list(a.shape)
    header = [len(header)] + header
    header = np.array(header).astype('int32')
    with open(fp, 'wb') as f:
        header.tofile(f)
        a.tofile(f)
    print(header)
    print(a.ravel())


def main():
    s = int(sys.argv[-1])
    a = np.random.normal(size=(s, s)).astype('float32')
    export_array(a, 'a.bin')
    b = np.random.normal(size=(s, s)).astype('float32')
    export_array(b, 'b.bin')
    c = np.dot(a, b)
    export_array(c, 'c.bin')


main()
