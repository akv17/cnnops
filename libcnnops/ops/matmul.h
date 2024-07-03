#include <stdlib.h>
#include "../tensor.h"


Tensor *matmul_cpu_float32(Tensor *a, Tensor *b);


float32_t _get(float *buffer, int i, int j, int col_stride) {
    int idx = i * col_stride + j;
    return buffer[idx];
}


void _set(float *buffer, int i, int j, int col_stride, float32_t v) {
    int idx = i * col_stride + j;
    buffer[i * col_stride + j] = v;
}


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

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float32_t c_acc = 0.0;
            for (int k = 0; k < common_dim; k++) {
                float32_t ai = _get(a_buf, i, k, a_col_str);
                float32_t bi = _get(b_buf, k, j, b_col_str);
                c_acc += ai * bi;
            }
            _set(c_buf, i, j, c_col_str, c_acc);
        }
    }

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
