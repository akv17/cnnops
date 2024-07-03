#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

typedef float float32_t;

typedef struct Tensor Tensor; 
struct Tensor {
    int32_t dtype;
    int32_t item_size;
    int32_t num_items;
    int32_t num_dims;
    int32_t *shape;
    void *buffer;
};

#endif
