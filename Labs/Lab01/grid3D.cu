#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"

/*
Declaring two large 3D arrays. They are allocated in the GPU memory, so they are usable by kernel but not by host functions.
Usually is not used because:
    - array dimension is set at compile time and not run time
    - declaring variables in file scope is deeply depreciated

Note that in C++ a 3D variable is defined as z,y,x while in CUDA is x,y,zn
*/
__device__  int   a[256][512][512];  // file scope
__device__  float b[256][512][512];  // file scope

__global__ void grid3D(int nx,int ny,int nz,int id)
{
    /*
    Compute the threads x,y and z coordinates within its threads block
    */
	int x = blockIdx.x*blockDim.x+threadIdx.x; // find (x,y,z) in
	int y = blockIdx.y*blockDim.y+threadIdx.y; // in arrays
	int z = blockIdx.z*blockDim.z+threadIdx.z; // 
	if(x >=nx || y >=ny || z >=nz) return;     // out of range. Good practice

    /*
    Computing some values
    */
	int array_size = nx*ny*nz;
	int block_size = blockDim.x*blockDim.y*blockDim.z;
	int grid_size  =  gridDim.x* gridDim.y* gridDim.z;
	int total_threads = block_size*grid_size;
	int thread_rank_in_block = (threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
	int block_rank_in_grid  =  (blockIdx.z*gridDim.y+ blockIdx.y)*gridDim.x+ blockIdx.x;
	int thread_rank_in_grid = thread_rank_in_block + block_size*block_rank_in_grid;

    /*
    Store some values into the array a and b using indices derived from the threads position in the 3D thread grid
    */
    a[z][y][x] = thread_rank_in_grid;
	b[z][y][x] = sqrtf((float)a[z][y][x]);

    /*
    For some thread chosen by the user, we print some of the computed values
    */
	if(thread_rank_in_grid == id) {
		printf("array size   %3d x %3d x %3d = %d\n",nx,ny,nz,array_size);
		printf("thread block %3d x %3d x %3d = %d\n",blockDim.x,blockDim.y,blockDim.z,block_size);
		printf("thread  grid %3d x %3d x %3d = %d\n",gridDim.x,gridDim.y,gridDim.z,grid_size);
		printf("total number of threads in grid %d\n",total_threads);
		printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",z,y,x,a[z][y][x],z,y,x,b[z][y][x]);
		printf("rank_in_block = %d rank_in_grid = %d rank of block_rank_in_grid = %d\n",thread_rank_in_block,thread_rank_in_grid,block_rank_in_grid);
	}
}

int main(int argc,char *argv[])
{
	int id = (argc > 1) ? atoi(argv[1]) : 12345;
	dim3 thread3d(32,8,2); // 32*8*2    = 512
	dim3  block3d(16,64,128); // 16*64*128 = 131072
	grid3D<<<block3d,thread3d>>>(512,512,256,id);
    cudaDeviceSynchronize();
	return 0;
}