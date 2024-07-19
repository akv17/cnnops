import numpy as np

from .binding import LibMatmul


def main():
    a = np.random.normal(size=(2, 4)).astype('float32')
    b = np.random.normal(size=(4, 8)).astype('float32')
    res_np = np.dot(a, b)
    
    lib = LibMatmul.load()
    spec = lib.create_op_spec(a, b)
    res_lib = lib.matmul(spec, a, b)

    print(res_np.ravel())
    print([*res_lib])


if __name__ == '__main__':
    main()
