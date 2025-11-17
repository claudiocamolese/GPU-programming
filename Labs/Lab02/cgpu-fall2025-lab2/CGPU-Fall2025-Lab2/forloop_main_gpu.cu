/*
This code implements a matrix multiplication using gpu for increasing size of the matrix to compare the results
*/

#include <cmath>
#include <iostream>
#include <string>

#include <cuda.h>
#include "args.hxx"

// Matrix manipulation function
#include "matrix_utils.h"

// Define different gemm kernel
#include "gemm_kernel.cuh"

#define REAL float
#define BLOCK_SIZE 32

int main(int argc, char **argv) {

    std::cout << "[Matrix Multiply Using CUDA] - Starting..." << std::endl;

    // Define parser 
    args::ArgumentParser parser("gemm_cuda", "Matrix Multiply using CUDA");

    // Set parser value
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> size_start_flag(parser, "size", "Initial square matrix size", {"start"}, 256);
    args::ValueFlag<int> size_max_flag(parser, "size", "Maximum square matrix size", {"max"}, 2048);

    // Invoke parser
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

    // Read initial and maximum size
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

    // Loop over increasing matrix sizes (square matrices, powers of 2)
    int n_iter = 0;
    for (int dim = size_start; dim <= size_max; dim *= 2, ++n_iter) {

        std::cout << "\n=== Iteration " << n_iter << ": Matrix size " << dim << "x" << dim << " ===\n";

        int WA = dim, HA = dim;
        int WB = dim, HB = dim;
        int WC = WA, HC = HB;

        // Allocate host memory
        unsigned int size_A = WA * HA;
        float *h_A = (float*) malloc(sizeof(float) * size_A);
        unsigned int size_B = WB * HB;
        float *h_B = (float*) malloc(sizeof(float) * size_B);
        float *h_C = (float*) malloc(sizeof(float) * (WA*HB));

        // Initialize matrices
        fill_random<REAL>(h_A, WA, HA);
        fill_random<REAL>(h_B, WB, HB);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, sizeof(float) * size_A);
        cudaMalloc((void**)&d_B, sizeof(float) * size_B);
        cudaMalloc((void**)&d_C, sizeof(float) * (WA*HB));

        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Copy host -> device
        cudaMemcpy(d_A, h_A, sizeof(float)*size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(float)*size_B, cudaMemcpyHostToDevice);

        // Setup execution configuration
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((WC + threads.x - 1)/threads.x, (HC + threads.y - 1)/threads.y);

        // Launch kernel
        cudaEventRecord(start, 0);
        gemm_naive<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Copy result back
        cudaMemcpy(h_C, d_C, sizeof(float)*(WA*HB), cudaMemcpyDeviceToHost);

        // Compute elapsed time
        float msecTotal = 0;
        cudaEventElapsedTime(&msecTotal, start, stop);

        // Compute performance
        float flop = 2.0f * WC * HC * WA;
        std::cout << "\tProcessing time: " << msecTotal << " ms\n";
        std::cout << "\tGFLOPS: " << flop / msecTotal / 1e6 << "\n";

        // Free memory
        free(h_A); free(h_B); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    return (EXIT_SUCCESS);
}
