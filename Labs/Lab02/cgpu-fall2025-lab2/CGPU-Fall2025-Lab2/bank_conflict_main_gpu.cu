#include <iostream>
#include <cuda.h>
#include "args.hxx"
#include "matrix_utils.h"                  // fill_random
#include "gemm_tiled_bankfree.cuh"         // kernel ottimizzato

#define REAL float
#define BLOCK_SIZE 32

// Trasposizione CPU
template <typename T>
void transpose_matrix(T* B, T* B_T, int rows, int cols) {
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            B_T[j*rows + i] = B[i*cols + j];
}

int main(int argc, char **argv) {

    std::cout << "[Tiled Matrix Multiply CUDA - Bank Conflict Free] - Starting...\n";

    args::ArgumentParser parser("gemm_cuda", "Tiled Matrix Multiply CUDA bank conflict free");
    args::HelpFlag help(parser, "help", "Display help menu", {'h',"help"});
    args::ValueFlag<int> size_start_flag(parser,"size","Initial square matrix size",{"start"},256);
    args::ValueFlag<int> size_max_flag(parser,"size","Maximum square matrix size",{"max"},2048);

    try { parser.ParseCLI(argc, argv); }
    catch(args::Help){ std::cout << parser; return 0;}
    catch(args::ParseError e){ std::cerr<<e.what()<<std::endl; std::cerr<<parser; return 1;}
    catch(args::ValidationError e){ std::cerr<<e.what()<<std::endl; std::cerr<<parser; return 1;}

    int size_start = args::get(size_start_flag);
    int size_max   = args::get(size_max_flag);

    // CUDA setup
    cudaDeviceProp deviceProp;
    int devID=0;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);
    std::cout << "GPU Device " << devID << ": \"" << deviceProp.name 
              << "\" compute capability " << deviceProp.major << "." << deviceProp.minor << "\n";

    // Loop potenze di 2
    int n_iter=0;
    for(int N=size_start; N<=size_max; N*=2, ++n_iter){
        std::cout<<"\n=== Iteration "<<n_iter<<": Matrix size "<<N<<"x"<<N<<" ===\n";
        unsigned int size_matrix = N*N;

        // Host memory
        REAL *h_A   = (REAL*) malloc(sizeof(REAL)*size_matrix);
        REAL *h_B   = (REAL*) malloc(sizeof(REAL)*size_matrix);
        REAL *h_B_T = (REAL*) malloc(sizeof(REAL)*size_matrix);
        REAL *h_C   = (REAL*) malloc(sizeof(REAL)*size_matrix);

        // Initialize
        fill_random<REAL>(h_A,N,N);
        fill_random<REAL>(h_B,N,N);

        // Trasposizione B CPU
        transpose_matrix<REAL>(h_B,h_B_T,N,N);

        // Device memory
        REAL *d_A,*d_B,*d_C;
        cudaMalloc((void**)&d_A,sizeof(REAL)*size_matrix);
        cudaMalloc((void**)&d_B,sizeof(REAL)*size_matrix);
        cudaMalloc((void**)&d_C,sizeof(REAL)*size_matrix);

        cudaMemcpy(d_A,h_A,sizeof(REAL)*size_matrix,cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,h_B_T,sizeof(REAL)*size_matrix,cudaMemcpyHostToDevice);

        // Timing
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
        dim3 grid((N+threads.x-1)/threads.x,(N+threads.y-1)/threads.y);

        cudaEventRecord(start,0);
        gemm_tiled_bankfree<<<grid,threads>>>(d_C,d_A,d_B,N);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaMemcpy(h_C,d_C,sizeof(REAL)*size_matrix,cudaMemcpyDeviceToHost);

        float msecTotal=0;
        cudaEventElapsedTime(&msecTotal,start,stop);

        float flop = 2.0f*N*N*N;
        std::cout<<"\tProcessing time: "<<msecTotal<<" ms\n";
        std::cout<<"\tGFLOPS: "<<flop/msecTotal/1e6<<"\n";

        // Cleanup
        free(h_A); free(h_B); free(h_B_T); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    return 0;
}
