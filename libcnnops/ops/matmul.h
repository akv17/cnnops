#include <stdlib.h>
#include "../tensor.h"
#include "../timer.h"

#define _idx(i, j, s) (i * s + j)


Tensor *matmul_cpu_float32(Tensor *a, Tensor *b);


Tensor *matmul_cpu_float32(Tensor *a, Tensor *b) {
    float32_t *a_buf = (float32_t *) a->buffer;
    float32_t *b_buf = (float32_t *) b->buffer;

    int32_t a_rows = a->shape[0];
    int32_t a_cols = a->shape[1];
    int32_t b_rows = b->shape[0];
    int32_t b_cols = b->shape[1];
    int32_t common_dim = a_cols;
    int32_t a_col_str = a_cols;
    int32_t b_col_str = b_cols;
    
    int32_t c_rows = a_rows;
    int32_t c_cols = b_cols;
    int32_t c_col_str = c_cols;
    int32_t c_size = c_rows * c_cols;
    float32_t *c_buf = (float32_t *) malloc(sizeof(float32_t) * c_size);

    _kernel0(
        a_buf,
        b_buf,
        c_buf,
        a_rows,
        a_col_str,
        b_cols,
        b_col_str,
        common_dim,
        c_col_str
    );

    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    int32_t *t_shape = (int32_t *) malloc(sizeof(int32_t) * 2);
    t_shape[0] = c_rows;
    t_shape[1] = c_cols;
    t->name = "c";
    t->dtype = 0;
    t->item_size = a->item_size;
    t->num_items = c_size;
    t->num_dims = 2;
    t->shape = t_shape;
    t->buffer = c_buf;
    return t;
}


void _kernel0(
    float32_t *a,
    float32_t *b,
    float32_t *c,
    int32_t a_rows,
    int32_t a_col_str,
    int32_t b_cols,
    int32_t b_col_str,
    int32_t common_dim,
    int32_t c_col_str
) {
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float32_t c_acc = 0.0;
            for (int k = 0; k < common_dim; k++) {
                // float32_t ai = a[_idx(i, k, a_col_str)];
                // float32_t bi = b[_idx(k, j, b_col_str)];
                // c_acc += ai * bi;
                // float32_t ai = a[_idx(i, k, a_col_str)];
                // float32_t bi = b[_idx(k, j, b_col_str)];
                c_acc += a[_idx(i, k, a_col_str)] * b[_idx(k, j, b_col_str)];
            }
            c[_idx(i, j, c_col_str)] = c_acc;
        }
    }
}


// void _kernel1(
//     float32_t *a,
//     float32_t *b,
//     float32_t *c,
//     int32_t a_rows,
//     int32_t a_col_str,
//     int32_t b_cols,
//     int32_t b_col_str,
//     int32_t common_dim,
//     int32_t c_col_str
// ) {
//     for (int i = 0; i < a_rows; i++) {
//         for (int k = 0; k < common_dim; k++) {
//             float32_t c_acc = 0.0;
//             for (int j = 0; j < b_cols; j++) {
//                 float32_t ai = a[_idx(i, k, a_col_str)];
//                 float32_t bi = b[_idx(k, j, b_col_str)];
//                 c_acc += ai * bi;
//             }
//             c[_idx(i, j, c_col_str)] = c_acc;
//         }
//     }
// }