/*
This code implements the trapezoidal rule to evaluate the integral of sin(x) from 0 to pi, 
based on the sum of a large umber of equally spaced evaluations of the function in this range
*/

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "._cxtimers.h"
#include<math.h>
#include<thrust/device_vector.h>

/*
__host__ and __device__ tells the compiler to do two versions of the function, one for the CPU and one for the GPU
*/
__host__ __device__ inline float sinsum(float x, int terms){
    float x2 = x*x;
    float term = x;
    float sum =  term;

    for(int n=1; n<terms; n++){
        term*= -x2 / (2*n*(2*n-1));
        sum += term;
    }

    return sum;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size){

    int step = blockIdx.x * blockDim.x + threadIdx.x;

    if(step<steps){
        // this is an out-of-range check. the kernel will exit at this point for threads that fail the check
        float x = step_size*step;
        sums[step] = sinsum(x, terms);
    }
}


int main(int argc, char const *argv[])
{
    
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;

    double pi = M_PI;
    double step_size = pi / (steps - 1);
    
    /* 
    this line creates the arrays of size steps in the device memory using the thrust device_vector class.
    All values are initialized as zeros on the device.
    This array is used by the kernel to hold the individual values returned by calls the simsym function.
    */
    thrust::device_vector<float> dsums(steps);

    /*
    We can't pass directly dsums to the kernel since it is not possible. So, we need to pass a pointer to the memory array.
    Then, we need to cast the pointer from cpu to device.
    
    Why you need to add [0]?
    Because &dsums[0] take the adress of the first element in the memory, so a pointer to the contigue buffer of the vector.
    In C++, a contigue vector in the memory can be treated as an array. All the other elements are consecutive to dims[0

    Why can't I use &dsums?
    &dsums don't point to the first element in the GPU but to the device_vector class. This is not compativle with the kernel CUDA
    
    Why I need to cast?
    Because otherwise it wouldn't work for device
    */

    float *dptr = thrust::raw_pointer_cast(&dsums[0]);

    cx::timer tim;

    // Launching the kernel
    gpu_sin<<<blocks, threads>>>(dptr, steps, terms, (float) step_size);

    // sum all the elements in the array dsums in GPU memory USING GPU
    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
    double gpu_time = tim.lap_ms();

    gpu_sum -= 0.5*(sinsum(0.0f, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;
    
    printf("gpusum %.10f steps %d terms %d time %.3f ms\n", gpu_sum, steps, terms, gpu_time);
    
    return 0;
}
