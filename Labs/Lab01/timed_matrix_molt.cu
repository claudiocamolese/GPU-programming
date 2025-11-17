#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

__global__ void product(float *A, float *B, float *C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col]; // A is coalascing, while B not
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 16;
    std::vector<float> gpu_time;
    std::vector<std::chrono::duration<double, std::milli>> cpu_time;

    while (N <= 1024) {
        std::cout << "\n=== Matrix size: " << N << "x" << N << " ===\n";
        int size = N * N * sizeof(float);
        float computing_gpu_time;

        // Allocazione CPU
        float *h_A = new float[N * N];
        float *h_B = new float[N * N];
        float *h_C = new float[N * N];

        // Inizializzazione
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                h_A[i * N + j] = 10 + i;
                h_B[i * N + j] = float(j) / N;
            }

        // ---- CPU multiplication ----
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k)
                    sum += h_A[i * N + k] * h_B[k * N + j];
                h_C[i * N + j] = sum;
            }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> computing_cpu_time = end_cpu - start_cpu;

        // ---- GPU implementation ----
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (N + blockSize.y - 1) / blockSize.y);

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        product<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

        // Controllo errori
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cout << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cout << "CUDA error during kernel execution: " << cudaGetErrorString(err) << std::endl;

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&computing_gpu_time, start, end);

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        std::cout << "CPU time: " << computing_cpu_time.count() << " ms\n";
        std::cout << "GPU time: " << computing_gpu_time << " ms\n";

        gpu_time.push_back(computing_gpu_time);
        cpu_time.push_back(computing_cpu_time);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;

        N *= 2;
    }

    std::cout << "\n=== Confronto CPU vs GPU ===\n";

    for (int i = 0; i < gpu_time.size(); i++) {
        float cpu_time_ms = static_cast<float>(cpu_time[i].count());

        std::cout << "N = " << (16 * (1 << i))  // perché N raddoppia ogni volta
              << " -> CPU: " << cpu_time_ms << " ms, "
              << "GPU: " << gpu_time[i] << " ms";

        if (gpu_time[i] < cpu_time_ms)
        std::cout << " GPU più veloce";
        else
            std::cout << " CPU più veloce";

        std::cout << "\n";
    }

    return 0;
}
