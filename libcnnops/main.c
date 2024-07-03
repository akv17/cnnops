#include "tensor.h"
#include "read.h"


int main() {
    Tensor *t = read_tensor("../a.bin", "x");
    print_tensor(t, -1);
}
