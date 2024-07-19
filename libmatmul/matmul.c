#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define _i2d(i, j, s) (i * s + j)


typedef struct OpSpec OpSpec;
struct OpSpec {
    int a_rows;
    int a_cols;
    int a_stride;

    int b_rows;
    int b_cols;
    int b_stride;

    int c_rows;
    int c_cols;
    int c_stride;
    int c_size;

    int batch_size;
    int batch_a_stride;
    int batch_b_stride;
    int batch_c_stride;
};


OpSpec create_op_spec(int a_nd, int b_nd, int* a_shape, int *b_shape) {
    int a_rows = a_shape[a_nd - 2];
    int a_cols = a_shape[a_nd - 1];
    int a_stride = a_cols;

    int b_rows = b_shape[b_nd - 2];
    int b_cols = b_shape[b_nd - 1];
    int b_stride = b_cols;

    int c_rows = a_rows;
    int c_cols = b_cols;
    int c_stride = c_cols;

    int batch_size = 1;
    for (int i = 0; i < a_nd - 2; i++) {
        batch_size *= a_shape[i];
    }
    int batch_a_stride = a_rows * a_cols;
    int batch_b_stride = b_rows * b_cols;
    int batch_c_stride = c_rows * c_cols;
    
    int c_size = c_rows * c_cols * batch_size;

    OpSpec spec = {
        a_rows,
        a_cols,
        a_stride,
        b_rows,
        b_cols,
        b_stride,
        c_rows,
        c_cols,
        c_stride,
        c_size,
        batch_size,
        batch_a_stride,
        batch_b_stride,
        batch_c_stride,
    };
    return spec;
}


void _matmul_naive(
    float *a,
    float *b,
    float *c,
    int ar,
    int ac,
    int as,
    int br,
    int bc,
    int bs,
    int cr,
    int cc,
    int cs
) {
    #pragma omp parallel for
    for (int i = 0; i < ar; i++) {
        for (int j = 0; j < bc; j++) {
            for (int k = 0; k < ac; k++) {
                c[_i2d(i, j, cs)] += a[_i2d(i, k, as)] * b[_i2d(k, j, bs)];
            }
        }
    }
}


void matmul(
    int kernel,
    OpSpec spec,
    float *a,
    float *b,
    float *c
) {
    _matmul_naive(
        a,
        b,
        c,
        spec.a_rows,
        spec.a_cols,
        spec.a_stride,
        spec.b_rows,
        spec.b_cols,
        spec.b_stride,
        spec.c_rows,
        spec.c_cols,
        spec.c_stride
    );
}
