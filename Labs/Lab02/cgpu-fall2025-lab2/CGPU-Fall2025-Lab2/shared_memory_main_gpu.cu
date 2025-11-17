// -----------------------------------------------------------------------------
// * Name:       main_gpu.cxx
// * Purpose:    Tiled Matrix multiplication on GPU
// * History:    Adapted for shared memory tiling
// -----------------------------------------------------------------------------

#include <iostream>
#include <string>
#include <cuda.h>
#include "args.hxx"           // command-line parser
#include "matrix_utils.h"     // funzione fill_random
#include "gemm_tiled_kernel.cuh"  // kernel tiled

#define REAL float
#define BLOCK_SIZE 32

int main(int argc, char **argv) {

    std::cout << "[Tiled Matrix Multiply Using CUDA] - Starting..." << std::endl;

    // Parser CLI
    args::ArgumentParser parser("gemm_cuda", "Tiled Matrix Multiply using CUDA");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> size_start_flag(parser, "size", "Initial square matrix size", {"start"}, 256);
    args::ValueFlag<int> size_max_flag(parser, "size", "Maximum square matrix size", {"max"}, 2048);

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    } catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    int size_start = args::get(size_start_flag);
    int size_max   = args::get(size_max_flag);

    // Setup CUDA environment
    cudaError_t error;
    cudaDeviceProp deviceProp;
    int devID = 0;
    error = cudaGetDevice(&devID);
    if (error != cudaSuccess) {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        return 1;
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        return 1;
    } else {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
               devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Loop su matrici quadrate con dimensioni potenze di 2
    int n_iter = 0;
    for (int dim = size_start; dim <= size_max; dim *= 2, ++n_iter) {

        std::cout << "\n=== Iteration " << n_iter << ": Matrix size " << dim << "x" << dim << " ===\n";

        int N = dim;

        // Allocate host memory
        unsigned int size_matrix = N * N;
        REAL *h_A = (REAL*) malloc(sizeof(REAL) * size_matrix);
        REAL *h_B = (REAL*) malloc(sizeof(REAL) * size_matrix);
        REAL *h_C = (REAL*) malloc(sizeof(REAL) * size_matrix);

        // Initialize matrices
        fill_random<REAL>(h_A, N, N);
        fill_random<REAL>(h_B, N, N);

        // Allocate device memory
        REAL *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, sizeof(REAL) * size_matrix);
        cudaMalloc((void**)&d_B, sizeof(REAL) * size_matrix);
        cudaMalloc((void**)&d_C, sizeof(REAL) * size_matrix);

        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Copy host -> device
        cudaMemcpy(d_A, h_A, sizeof(REAL)*size_matrix, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(REAL)*size_matrix, cudaMemcpyHostToDevice);

        // Setup execution configuration
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

        // Launch tiled kernel
        cudaEventRecord(start, 0);
        gemm_tiled<<<grid, threads>>>(d_C, d_A, d_B, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Copy result back
        cudaMemcpy(h_C, d_C, sizeof(REAL)*size_matrix, cudaMemcpyDeviceToHost);

        // Compute elapsed time
        float msecTotal = 0;
        cudaEventElapsedTime(&msecTotal, start, stop);

        // Compute performance
        float flop = 2.0f * N * N * N;
        std::cout << "\tProcessing time: " << msecTotal << " ms\n";
        std::cout << "\tGFLOPS: " << flop / msecTotal / 1e6 << "\n";

        // Free memory
        free(h_A); free(h_B); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    return 0;
}
