#include <stdio.h>


#include "tensor.h"
#include "read.h"
#include "timer.h"
#include "ops/matmul.h"

#define TENSOR_BUFFER_PRINT_LIMIT 16


int main() {
    Tensor *a = read_tensor("a.bin", "a");
    Tensor *b = read_tensor("b.bin", "b");
    Tensor *c_in = read_tensor("c.bin", "c_in");
    print_tensor(a, TENSOR_BUFFER_PRINT_LIMIT);
    print_tensor(b, TENSOR_BUFFER_PRINT_LIMIT);
    print_tensor(c_in, TENSOR_BUFFER_PRINT_LIMIT);

    timer_start();
    Tensor *c_out = matmul_cpu_float32(a, b);
    timer_end();
    
    // print_tensor(c_out, TENSOR_BUFFER_PRINT_LIMIT);

    timer_print();
}
