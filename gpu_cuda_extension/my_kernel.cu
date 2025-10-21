// my_kernel.cu
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Lancia il kernel
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                    b.data_ptr<float>(),
                                    c.data_ptr<float>(),
                                    N);

    // Sincronizza e controlla errori
    cudaDeviceSynchronize();
}
