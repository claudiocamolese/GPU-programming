#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>

// Kernel CUDA che somma due tensori
__global__ void add_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

// Funzione chiamabile da wrapper
void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int N = a.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n",
