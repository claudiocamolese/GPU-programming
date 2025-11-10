#include <torch/extension.h>

// Dichiarazione della funzione CUDA (definita in my_kernel.cu)
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

// ==================== PYBIND11 BINDINGS ====================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Add two tensors (CUDA)");
}
