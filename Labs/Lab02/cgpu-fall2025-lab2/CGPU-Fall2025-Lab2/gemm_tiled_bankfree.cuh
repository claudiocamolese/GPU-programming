#pragma once
#include <cuda.h>

#define BLOCK_SIZE 32
#define REAL float

// Kernel tiled con shared memory senza bank conflict
__global__ void gemm_tiled_bankfree(REAL *C, const REAL *A, const REAL *B, int N) {

    __shared__ REAL As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ REAL Bs[BLOCK_SIZE][BLOCK_SIZE+1]; // +1 per evitare bank conflict

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    REAL Cvalue = 0;

    for (int t = 0; t < (N + BLOCK_SIZE - 1)/BLOCK_SIZE; ++t) {

        // Carica tile di A
        if(row < N && t*BLOCK_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row*N + t*BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Carica tile di B trasposto in shared memory per evitare bank conflict
        if(col < N && t*BLOCK_SIZE + threadIdx.y < N)
            Bs[threadIdx.x][threadIdx.y] = B[(t*BLOCK_SIZE + threadIdx.y)*N + col];
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Moltiplicazione dei tile
        for(int k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if(row < N && col < N)
        C[row*N + col] = Cvalue;
}
