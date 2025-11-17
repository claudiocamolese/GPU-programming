/*
Reduce algorithm 1: Parallel sum of N numbers:
    - Use N/2 threads to get N/2 pairwise sums
    - Set N=N/2 and iterate till N=1

To improve:
    - in x[t] += x[t + m]; we need to load both x[tid] and x[tid+m] into GPU registers and then storing it in x[tid]
        If we could accumulate partials sums in local registers, that would reudce the number of global memory access, which offers a speed up.
    
    - the host calls the kernel iteatively,halving the array size at each step and leaving the sum in the first array element.
        If we could instead perform the iteration inside the kernel that could also reduce the number of memory accesses required.
    
    - the array must be a multiple of 2
*/

#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include<math.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <random>
#include <chrono>


__global__ void reduce(float * x, int m){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    x[t] += x[t + m];
}



int main(int argc, char const *argv[])
{
    int N = (argc >1) ? atoi(argv[1]) : 1<<24;
    
    thrust::host_vector<float> host_x(N);
    thrust::device_vector<float> dev_x(N);

    // random initialization of the vector
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution unif(0.0, 1.0);

    for (int i = 0; i < N; i++) host_x[i] = unif(gen);

    dev_x = host_x;
    
    auto start1 = std::chrono::high_resolution_clock::now();
    double host_sum = 0.0;
    for(int i =0; i<N; i++) host_sum += host_x[i];
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t1 = end1 - start1;

    /*
    Reducing algorithm in GPU
    */
   cudaEvent_t start2, stop2;
   float ms;
   cudaEventCreate(&start2);
   cudaEventCreate(&stop2);
   
   cudaEventRecord(start2);

   for(int m = N/2; m > 0; m /= 2){
    /*
    m thread because there are m sum to compute but there is a max number of threads for blocks (1024 or 256 in this case).
    if m >256, take 256.
    if m <256, take m since is secure as number.
    */
    int threads = std::min(256,m);
    int blocks = std::max(m/256,1);

    reduce<<<blocks, threads>>>(dev_x.data().get(), m);
   }

    cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&ms, start2, stop2);
    double gpu_sum = dev_x[0];

    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1.count(), gpu_sum, ms);

    return 0;
}
