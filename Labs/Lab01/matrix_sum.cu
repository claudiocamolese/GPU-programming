/*Write a CUDA program that performs the matrix multiplication, in parallel ! Start with a grid size of (1, 1) and a
block size of (N ,N). Then try a bigger grid, with more blocks.
We must calculate an index from the thread and block numbers. That can look like this (in 1D) 
*/
#include <iostream>


__global__ void product(float *a, float *b, float *c, int N){

    // compute global indexing in 2D
    int row = threadIdx.y + blockIdx.y* blockDim.y; 
    int col = threadIdx.x + blockIdx.x* blockDim.x;
    
    // the condition is used to check that threads outside NxN cant accede the memory invalid memory.
    if(row<N && col <N){
        int index = row + col * N; // column major
        c[index] = a[index] + b[index];
    }
}

int main(int argc, char const *argv[])
{
    const int N = 16;
    const int size = N*N*sizeof(float);

    float *h_a = new float[N*N];
    float *h_b = new float[N*N];
    float *h_c = new float[N*N];

    // Initialize matrices
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            h_a[i+j*N] = 10 + i;
            h_b[i+j*N] = float(j)/N;

        }
    }

    // GPU allocations
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(N,N);
    dim3 gridSize(1,1);

    product<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    // copy results into the host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //printing
    std::cout<<"Matricx A: \n"<<std::endl;

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            std::cout << h_a[i+j*N] << " ";
        std::cout << "\n";
    }

    std::cout<<"Matricx B: \n"<<std::endl;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            std::cout << h_b[i+j*N] << " ";
        std::cout << "\n";
    }

    std::cout<<"Matricx C: \n"<<std::endl;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            std::cout << h_c[i+j*N] << " ";
        std::cout << "\n";
    }

    cudaFree(d_a), cudaFree(d_b), cudaFree(d_c);
    delete [] h_a, delete [] h_b, delete [] h_c;

    return 0;
}
