#include <torch/extension.h>
#include <vector>

// Dichiarazione kernel
void add_kernel(const float* a, const float* b, float* c, int n);

// Funzione wrapper per PyTorch
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    int n = a.numel();
    auto c = torch::zeros_like(a);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Chiamata kernel CUDA
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    cudaDeviceSynchronize();
    return c;
}

// Binding Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Add two tensors (CUDA)");
}
