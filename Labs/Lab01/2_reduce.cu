/*
This code improves 1_reduce because it implements the shared memory.
In this way,  the threads in each threads block can sum its individual accumulated totals and then write a single word with the block-sum to external memory
*/

#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include<math.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <random>
#include <chrono>


__global__ void reduce2(float *y,float *x,int N)
{
	extern __shared__ float tsum[]; // Dynamically Allocated Shared Mem
	int id = threadIdx.x;
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	tsum[id] = 0.0f;
	for(int k=tid;k<N;k+=stride) tsum[id] += x[k];
	__syncthreads();
	for(int k=blockDim.x/2; k>0; k /= 2){ // power of 2 reduction loop
		if(id<k) tsum[id] += tsum[id+k];
		__syncthreads();
	}
	if(id==0) y[blockIdx.x] = tsum[0]; // store one value per block
}

int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;  // power of 2
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	thrust::device_vector<float>  dy(blocks);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0); 

	for(int k = 0; k<N; k++) x[k] = fran(gen);

	dx = x;  // H2D copy (N words)
	auto start1 = std::chrono::high_resolution_clock::now();
	double host_sum = 0.0;

	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!

	auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t1 = end1 - start1;

	// simple GPU reduce for any value of N
	double gpu_sum = 0.0;
    auto start2 = std::chrono::high_resolution_clock::now();

	for(int i=0; i < nreps ;i++){
		reduce2<<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
		reduce2<<<1, blocks, blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks);

		if(i==0)  gpu_sum = dx[0];  
	}
	cudaDeviceSynchronize();
	auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t2 = end2 - start2;
	//double gpu_sum = dx[0]/nreps;          // D2H copy (1 word) 
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n", N, host_sum, t1.count(), gpu_sum, t2.count());
	return 0;
}
