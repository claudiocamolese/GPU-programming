/*
Reduce algorithm 1: Parallel sum of N numbers:
    - Use N/2 threads to get N/2 pairwise sums
    - Set N=N/2 and iterate till N=1
*/

#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "._cxtimers.h"
#include<math.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <random>


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
    
    cx::timer tim;
    double host_sum = 0.0;
    for(int i =0; i<N; i++) host_sum += host_x[i];
    double t1 = tim.lap_ms();

    tim.reset();

    /*
    Reducing algorithm in GPU
    */
   for(int m = N/2; m > 0; m /= 2){
    int threads = std::min(256,m);
    int blocks = std::max(m/256,1);

    reduce<<<blocks, threads>>>(dev_x.data().get(), m);
    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();
    double gpu_sum = dev_x[0];

    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1, gpu_sum, t2);
   }
    return 0;
}
