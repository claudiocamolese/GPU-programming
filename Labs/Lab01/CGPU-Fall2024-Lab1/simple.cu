// Assigns every element in an array with its index.
// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <iostream>
#include <math.h>

__global__ void simple(float *c) 
{
	c[threadIdx.x] = threadIdx.x;
}

int main()
{
	// Define problem size
	const int N = 16;
	
	// Define number of blocks
	const int blocksize = 16; 

	// Create host and device data strutures
	float *c_h = new float[N];	
	float *c_d; // CPU cant allocate memory directly with new, you have to use malloc.	 

	// Give size of array to allocate on GPU
	const int size = N*sizeof(float);
	
	//	Allocate array on GP GPU
	cudaMalloc( (void**)&c_d, size );

	// Define workspace topology
	dim3 dimBlock( blocksize, 1 ); // dim3 nome_blocco(x_threads, y_threads=1, z_threads=1); here we have 16 threads on x, 1 one y and 1 on z.

	dim3 dimGrid( 1, 1 );  //dim3 dimGrid(x_blocks, y_blocks=1, z_blocks=1);; here we have 1 blocks on x, 1 blocks on y and 1 blocks on z.

	// Execute kernel, totale_thread=16∗1=16
	simple<<<dimGrid, dimBlock>>>(c_d);

	// Wait for kernel completion
	cudaDeviceSynchronize();

	// Copy result of computation back on host
	cudaMemcpy( c_h, c_d, size, cudaMemcpyDeviceToHost ); 
	
	for (int i = 0; i < N; i++)
		std::cout<<c_h[i]<<" ";
		
	std::cout<<std::endl;

	// Free memory
	cudaFree( c_d );
	delete[] c_h;
	
	std::cout<<"done"<<std::endl;
	
	return EXIT_SUCCESS;
}
