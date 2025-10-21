// my_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int N = a.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Lancia il kernel
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                    b.data_ptr<float>(),
                                    c.data_ptr<float>(),
                                    N);

    // Sincronizza e controlla errori
    cudaDeviceSynchronize();
}
