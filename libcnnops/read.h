#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "tensor.h"

typedef struct Header Header; 
struct Header {
    int32_t dtype;
    int32_t item_size;
    int32_t num_items;
    int32_t num_dims;
    int32_t *shape;
};
Tensor *read_tensor(char *path);


Header *_read_header(FILE *fp) {
    // header: header_size, dtype, item_size, num_items, ndim, *shape
    
    int32_t header_size_buf[1];
    fread(header_size_buf, sizeof(int32_t), 1, fp);
    int32_t header_size = header_size_buf[0];
    
    int32_t header_buf[header_size];
    fread(header_buf, sizeof(int32_t), header_size, fp);

    int32_t dtype = header_buf[0];
    int32_t item_size = header_buf[1];
    int32_t num_items = header_buf[2];
    int32_t num_dims = header_buf[3];

    int32_t *shape = (int32_t *) malloc(sizeof(int32_t) * num_dims);
    int j = 0;
    for (int i = 4; i < header_size; i++, j++) {
        shape[j] = header_buf[i];
    }

    Header *h = (Header *) malloc(sizeof(Header));
    h->dtype = dtype;
    h->item_size = item_size;
    h->num_items = num_items;
    h->num_dims = num_dims;
    h->shape = shape;
    return h;
}

void *_read_data(FILE *fp, int32_t item_size, int32_t num_items) {
    void *buf = malloc(item_size * num_items);
    fread(buf, item_size, num_items, fp);
    return buf;
}


void _print_header(Header *h) {
    printf("dtype: %d\n", h->dtype);
    printf("item_size: %d\n", h->item_size);
    printf("num_items: %d\n", h->num_items);
    printf("num_dims: %d\n", h->num_dims);
    printf("shape: (");
    for (int i = 0; i < h->num_dims; i++) {
        printf("%d", h->shape[i]);
        if (i < h->num_dims - 1) {
            printf(", ");
        }
    }
    printf(")");
    printf("\n");
}


Tensor *read_tensor(char *path) {
    FILE *fp;
    fp = fopen(path, "rb");
    Header *h = _read_header(fp);
    float *data = (float *) _read_data(fp, h->item_size, h->num_items);
    fclose(fp);
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    t->dtype = h->dtype;
    t->item_size = h->item_size;
    t->num_items = h->num_items;
    t->num_dims = h->num_dims;
    t->shape = h->shape;
    t->buffer = data;
    return t;
}
