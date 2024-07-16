#include <stdio.h>

#include "tensor.h"
#include "read.h"
#include "timer.h"
#include "ops/matmul.h"

#define TENSOR_BUFFER_PRINT_LIMIT 8


int main() {
    Tensor *a = read_tensor("a.bin", "a");
    Tensor *b = read_tensor("b.bin", "b");
    Tensor *c_in = read_tensor("c.bin", "c_in");
    print_tensor(a, TENSOR_BUFFER_PRINT_LIMIT); printf("\n");
    print_tensor(b, TENSOR_BUFFER_PRINT_LIMIT); printf("\n");
    print_tensor(c_in, TENSOR_BUFFER_PRINT_LIMIT); printf("\n");

    Tensor *c_out = matmul(a, b);
    
    print_tensor(c_out, TENSOR_BUFFER_PRINT_LIMIT); printf("\n");
    compare_tensors(c_in, c_out, 0);
}
