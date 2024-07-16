#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
void compare_tensors(Tensor *a, Tensor *b);


void _print_tensor_buffer_float32(void *buffer, int buffer_size, int print_size) {
    int half_print_size = (int) print_size / 2;
    float *fbuffer = (float *) buffer;
    for (int i = 0; i < half_print_size; i++) {
        printf("%f", fbuffer[i]);
        if (i < half_print_size - 1) {
            printf(", ");
        }
    }
    printf(" ... ");
    for (int i = buffer_size - half_print_size; i < buffer_size; i++) {
        printf("%f", fbuffer[i]);
        if (i < buffer_size - 1) {
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
    _print_tensor_buffer_float32(t->buffer, t->num_items, buffer_size_print);
    printf("]");
    printf("\n");
}


void compare_tensors(Tensor *a, Tensor *b) {
    if (a->num_items != b->num_items) {
        printf("Size mismatch: %d, %d\n", a->num_items, b->num_items);
        exit(1);
    }
    float *a_buf = (float *) a->buffer;
    float *b_buf = (float *) b->buffer;
    int size = a->num_items;
    int has_diff = 0;
    for (int i = 0; i < size; i++) {
        float diff = (float) fabs(a_buf[i] - b_buf[i]);
        if (diff > 1e-5) {
            printf("Compare Diff at %d: %f %f %f\n", i, a_buf[i], b_buf[i], diff);
            has_diff = 1;
        }
    }
    if (!has_diff) {
        printf("Compare: OK\n");
    }
    else {
        printf("Compare: FAIL\n");
    }
}

#endif
