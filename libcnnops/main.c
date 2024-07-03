#include "tensor.h"
#include "read.h"
#include "ops/matmul.h"


int main() {
    Tensor *a = read_tensor("../a.bin", "a");
    Tensor *b = read_tensor("../b.bin", "b");
    Tensor *c_in = read_tensor("../c.bin", "c_in");
    print_tensor(a, -1);
    print_tensor(b, -1);
    print_tensor(c_in, -1);

    Tensor *c_out = matmul_cpu_float32(a, b);
    print_tensor(c_out, -1);
}
