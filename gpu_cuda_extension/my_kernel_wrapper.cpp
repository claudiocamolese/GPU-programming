#include <torch/extension.h>

// Dichiara la funzione definita nel .cu
void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// Binding PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Add two tensors on CUDA");
}
