#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdio.h>

typedef float float32_t;

typedef struct Tensor Tensor; 
struct Tensor {
    char *name;
    int32_t dtype;
    int32_t item_size;
    int32_t num_items;
    int32_t num_dims;
    int32_t *shape;
    void *buffer;
};
void print_tensor(Tensor *t, int buffer_limit);


void _print_tensor_buffer_float32(void *buffer, int size) {
    float *fbuffer = (float *) buffer;
    for (int i = 0; i < size; i++) {
        printf("%f", fbuffer[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
}


void print_tensor(Tensor *t, int buffer_size_limit) {
    printf("name: %s\n", t->name);
    printf("dtype: %d\n", t->dtype);
    printf("item_size: %d\n", t->item_size);
    printf("num_items: %d\n", t->num_items);
    printf("num_dims: %d\n", t->num_dims);
    printf("shape: (");
    for (int i = 0; i < t->num_dims; i++) {
        printf("%d", t->shape[i]);
        if (i < t->num_dims - 1) {
            printf(", ");
        }
    }
    printf(")");
    printf("\n");
    int buffer_size_print = buffer_size_limit == -1 ? t->num_items : buffer_size_limit;
    printf("data:\n");
    printf("[");
    _print_tensor_buffer_float32(t->buffer, buffer_size_print);
    printf("]");
    printf("\n");
}

#endif
