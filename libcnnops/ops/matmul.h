#include <stdlib.h>
#include <omp.h>

#include "../tensor.h"
#include "../timer.h"

#define _idx2d(i, j, s) (i * s + j)


Tensor *matmul(Tensor *a, Tensor *b);

typedef struct OpSpec OpSpec;
struct OpSpec {
    int32_t ndim;
    int32_t item_size;
    int32_t *shape;

    int32_t a_rows;
    int32_t a_cols;
    int32_t a_stride;
    
    int32_t b_rows;
    int32_t b_cols;
    int32_t b_stride;
    
    int32_t c_rows;
    int32_t c_cols;
    int32_t c_stride;
    int32_t c_size;
    
    int32_t batch_dims;
    int32_t batch_size;
    int32_t batch_stride;
};


OpSpec *_compute_op_spec(Tensor *a, Tensor *b) {
    int32_t ndim = a->num_dims;
    int32_t a_rows = a->shape[ndim-2];
    int32_t a_cols = a->shape[ndim-1];
    int32_t a_stride = a_cols;
    int32_t b_rows = b->shape[ndim-2];
    int32_t b_cols = b->shape[ndim-1];
    int32_t b_stride = b_cols;

    // fold into a batch all the dims up to the last two dims (rows and cols)
    int32_t batch_dims = ndim - 2;
    int32_t batch_size = 1;
    for (size_t i = 0; i < batch_dims; i++) {
        batch_size *= a->shape[i];
    }

    int32_t c_rows = a_rows;
    int32_t c_cols = b_cols;
    int32_t c_stride = c_cols;
    int32_t c_size = c_rows * c_cols * batch_size;
    int32_t batch_stride = c_rows * c_cols;

    int32_t *c_shape = (int32_t *) malloc(sizeof(int32_t) * ndim);
    for (size_t i = 0; i < ndim; i++) {
        c_shape[i] = a->shape[i];
    }
    c_shape[ndim-2] = c_rows;
    c_shape[ndim-1] = c_cols;

    OpSpec *spec = (OpSpec *) malloc(sizeof(OpSpec));
    spec->ndim = ndim;
    spec->item_size = a->item_size;
    spec->shape = c_shape;
    
    spec->a_rows = a_rows;
    spec->a_cols = a_cols;
    spec->a_stride = a_stride;
    
    spec->b_rows = b_rows;
    spec->b_cols = b_cols;
    spec->b_stride = b_stride;

    spec->c_rows = c_rows;
    spec->c_cols = c_cols;
    spec->c_stride = c_stride;
    spec->c_size = c_size;

    spec->batch_dims = batch_dims;
    spec->batch_size = batch_size;
    spec->batch_stride = batch_stride;
    return spec;
}


Tensor *_create_output_tensor(OpSpec *spec, float32_t *buffer) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    t->name = "c";
    t->dtype = 0;
    t->item_size = spec->item_size;
    t->num_items = spec->c_size;
    t->num_dims = spec->ndim;
    t->shape = spec->shape;
    t->buffer = buffer;
    return t;
}


void _kernel(
    int32_t offset,
    float32_t *a,
    float32_t *b,
    float32_t *c,
    int32_t a_rows,
    int32_t a_stride,
    int32_t b_cols,
    int32_t b_stride,
    int32_t common_dim,
    int32_t c_stride
) {
    #pragma omp parallel for
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float32_t c_acc = 0.0;
            for (int k = 0; k < common_dim; k++) {
                c_acc += a[_idx2d(i, k, a_stride) + offset] * b[_idx2d(k, j, b_stride) + offset];
            }
            c[_idx2d(i, j, c_stride) + offset] = c_acc;
        }
    }
}


// equivalent to PyTorch batch mamtmul supporting arbitrary number of dimensions.
// essentially this is just a series of 'k' matrix multiplications.
// all dims except last two (rows and cols) are folded into 'k' (total number of matrices to multiply). 
// each such k-th matrix is located via offset into 1d buffer. 
Tensor *matmul(Tensor *a, Tensor *b) {
    OpSpec *spec = _compute_op_spec(a, b);
    float32_t *a_buf = (float32_t *) a->buffer;
    float32_t *b_buf = (float32_t *) b->buffer;
    float32_t *c_buf = (float32_t *) malloc(sizeof(float32_t) * spec->c_size);
    int32_t batch_stride = spec->batch_stride;
    for (int32_t offset = 0; offset < spec->batch_stride; offset += batch_stride) {
        _kernel(
            offset,
            a_buf,
            b_buf,
            c_buf,
            spec->a_rows,
            spec->a_stride,
            spec->b_cols,
            spec->b_stride,
            spec->a_cols,
            spec->c_stride
        );
    }
    Tensor *out = _create_output_tensor(spec, c_buf);
    return out;
}
