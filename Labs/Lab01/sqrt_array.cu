#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>


__global__ void gpu_sqrt(float *arr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        arr[idx] = sqrtf(arr[idx]);
    }
}

int main() {
    const int N = 1 << 20; // 1 milione di elementi
    const int SIZE = N * sizeof(float);

    // Alloca array CPU
    float *h_input = new float[N];
    float *h_output_cpu = new float[N];
    float *h_output_gpu = new float[N];

    // Inizializza array
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // ------------------ CPU ------------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_output_cpu[i] = std::sqrt(h_input[i]);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // ------------------ GPU ------------------
    float *d_array;
    cudaMalloc((void**)&d_array, SIZE);
    cudaMemcpy(d_array, h_input, SIZE, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize; //formula to use all the threads

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_sqrt<<<gridSize, blockSize>>>(d_array, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_output_gpu, d_array, SIZE, cudaMemcpyDeviceToHost);

    // ------------------ Stampa risultati ------------------
    std::cout << "Primi 10 risultati CPU vs GPU:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "CPU: " << h_output_cpu[i] 
                  << " | GPU: " << h_output_gpu[i] << "\n";
    }

    std::cout << "\nTempo CPU: " << cpu_time.count() << " ms\n";
    std::cout << "Tempo GPU: " << gpu_time << " ms\n";

    // Cleanup
    cudaFree(d_array);
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    return 0;
}
