#include <cuda_runtime.h>
#include <torch/extension.h>

// ==================== CUDA KERNEL ====================
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ==================== CUDA WRAPPER ====================
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    int n = a.numel();
    auto c = torch::zeros_like(a);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Chiamata al kernel
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    cudaDeviceSynchronize();
    return c;
}
