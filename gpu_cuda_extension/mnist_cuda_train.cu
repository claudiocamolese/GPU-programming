#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define CONV1_FILTERS 32
#define CONV1_SIZE 5
#define POOL1_SIZE 2
#define CONV2_FILTERS 64
#define CONV2_SIZE 5
#define POOL2_SIZE 2
#define FC1_SIZE 128
#define OUTPUT_SIZE 10

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ===================== KERNEL CUDA =====================

__global__ void conv2d_kernel(float* input, float* output, float* weights, float* bias,
                               int batch_size, int in_channels, int out_channels,
                               int in_h, int in_w, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    int total = batch_size * out_channels * out_h * out_w;
    
    if (idx < total) {
        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int c_out = (idx / (out_w * out_h)) % out_channels;
        int b = idx / (out_w * out_h * out_channels);
        
        float sum = bias[c_out];
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_idx = b * in_channels * in_h * in_w +
                                c_in * in_h * in_w +
                                (h + kh) * in_w +
                                (w + kw);
                    int w_idx = c_out * in_channels * kernel_size * kernel_size +
                               c_in * kernel_size * kernel_size +
                               kh * kernel_size + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
        output[idx] = fmaxf(0.0f, sum); // ReLU
    }
}

__global__ void maxpool2d_kernel(float* input, float* output, int batch_size, int channels,
                                  int in_h, int in_w, int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    int total = batch_size * channels * out_h * out_w;
    
    if (idx < total) {
        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % channels;
        int b = idx / (out_w * out_h * channels);
        
        float max_val = -1e9;
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int in_idx = b * channels * in_h * in_w +
                            c * in_h * in_w +
                            (h * pool_size + ph) * in_w +
                            (w * pool_size + pw);
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        output[idx] = max_val;
    }
}

__global__ void fc_forward_kernel(float* input, float* output, float* weights, float* bias,
                                   int batch_size, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_size;
    
    if (idx < total) {
        int out_idx = idx % out_size;
        int b = idx / out_size;
        
        float sum = bias[out_idx];
        for (int i = 0; i < in_size; i++) {
            sum += input[b * in_size + i] * weights[out_idx * in_size + i];
        }
        output[idx] = sum;
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void softmax_kernel(float* input, float* output, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        __shared__ float max_val;
        __shared__ float sum_exp;
        
        if (threadIdx.x == 0) {
            max_val = input[b * size];
            for (int i = 1; i < size; i++) {
                max_val = fmaxf(max_val, input[b * size + i]);
            }
            
            sum_exp = 0.0f;
            for (int i = 0; i < size; i++) {
                sum_exp += expf(input[b * size + i] - max_val);
            }
        }
        __syncthreads();
        
        int idx = threadIdx.x;
        if (idx < size) {
            output[b * size + idx] = expf(input[b * size + idx] - max_val) / sum_exp;
        }
    }
}

// ===================== HOST FUNCTIONS =====================

void init_weights(float* weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

float compute_accuracy(float* predictions, int* labels, int size, int num_classes) {
    int correct = 0;
    for (int i = 0; i < size; i++) {
        int pred_class = 0;
        float max_prob = predictions[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            if (predictions[i * num_classes + j] > max_prob) {
                max_prob = predictions[i * num_classes + j];
                pred_class = j;
            }
        }
        if (pred_class == labels[i]) correct++;
    }
    return (float)correct / size;
}

extern "C" {
    void train_model(float* train_images, int* train_labels, int train_size,
                    float* test_images, int* test_labels, int test_size,
                    int epochs, int batch_size, float learning_rate) {
        
        srand(time(NULL));
        
        // Alloca e inizializza pesi
        int conv1_w_size = CONV1_FILTERS * 1 * CONV1_SIZE * CONV1_SIZE;
        int conv2_w_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE;
        int fc1_w_size = FC1_SIZE * (CONV2_FILTERS * 4 * 4);
        int fc2_w_size = OUTPUT_SIZE * FC1_SIZE;
        
        float *h_conv1_w, *h_conv1_b, *h_conv2_w, *h_conv2_b;
        float *h_fc1_w, *h_fc1_b, *h_fc2_w, *h_fc2_b;
        
        h_conv1_w = (float*)malloc(conv1_w_size * sizeof(float));
        h_conv1_b = (float*)malloc(CONV1_FILTERS * sizeof(float));
        h_conv2_w = (float*)malloc(conv2_w_size * sizeof(float));
        h_conv2_b = (float*)malloc(CONV2_FILTERS * sizeof(float));
        h_fc1_w = (float*)malloc(fc1_w_size * sizeof(float));
        h_fc1_b = (float*)malloc(FC1_SIZE * sizeof(float));
        h_fc2_w = (float*)malloc(fc2_w_size * sizeof(float));
        h_fc2_b = (float*)malloc(OUTPUT_SIZE * sizeof(float));
        
        init_weights(h_conv1_w, conv1_w_size);
        init_weights(h_conv2_w, conv2_w_size);
        init_weights(h_fc1_w, fc1_w_size);
        init_weights(h_fc2_w, fc2_w_size);
        
        for (int i = 0; i < CONV1_FILTERS; i++) h_conv1_b[i] = 0.0f;
        for (int i = 0; i < CONV2_FILTERS; i++) h_conv2_b[i] = 0.0f;
        for (int i = 0; i < FC1_SIZE; i++) h_fc1_b[i] = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) h_fc2_b[i] = 0.0f;
        
        // Alloca memoria GPU
        float *d_input, *d_conv1_out, *d_pool1_out, *d_conv2_out, *d_pool2_out;
        float *d_fc1_out, *d_output;
        float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
        float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b;
        
        CUDA_CHECK(cudaMalloc(&d_input, batch_size * 784 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_out, batch_size * CONV1_FILTERS * 24 * 24 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool1_out, batch_size * CONV1_FILTERS * 12 * 12 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_out, batch_size * CONV2_FILTERS * 8 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool2_out, batch_size * CONV2_FILTERS * 4 * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc1_out, batch_size * FC1_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, batch_size * OUTPUT_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_conv1_w, conv1_w_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_b, CONV1_FILTERS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_w, conv2_w_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_b, CONV2_FILTERS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc1_w, fc1_w_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc1_b, FC1_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc2_w, fc2_w_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fc2_b, OUTPUT_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_conv1_w, h_conv1_w, conv1_w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv1_b, h_conv1_b, CONV1_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv2_w, h_conv2_w, conv2_w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv2_b, h_conv2_b, CONV2_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fc1_w, h_fc1_w, fc1_w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fc1_b, h_fc1_b, FC1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fc2_w, h_fc2_w, fc2_w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fc2_b, h_fc2_b, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        printf("\n🚀 Inizializzazione completata - Inizio training su GPU\n\n");
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            printf("Epoch %d/%d\n", epoch + 1, epochs);
            
            int num_batches = (train_size + batch_size - 1) / batch_size;
            
            for (int batch = 0; batch < num_batches; batch++) {
                int start = batch * batch_size;
                int end = (start + batch_size < train_size) ? start + batch_size : train_size;
                int current_batch_size = end - start;
                
                // Copia batch su GPU
                CUDA_CHECK(cudaMemcpy(d_input, train_images + start * 784, 
                                     current_batch_size * 784 * sizeof(float), 
                                     cudaMemcpyHostToDevice));
                
                // Forward pass
                int threads = 256;
                int blocks;
                
                // Conv1
                blocks = (current_batch_size * CONV1_FILTERS * 24 * 24 + threads - 1) / threads;
                conv2d_kernel<<<blocks, threads>>>(d_input, d_conv1_out, d_conv1_w, d_conv1_b,
                                                   current_batch_size, 1, CONV1_FILTERS, 28, 28, 5);
                
                // Pool1
                blocks = (current_batch_size * CONV1_FILTERS * 12 * 12 + threads - 1) / threads;
                maxpool2d_kernel<<<blocks, threads>>>(d_conv1_out, d_pool1_out, 
                                                      current_batch_size, CONV1_FILTERS, 24, 24, 2);
                
                // Conv2
                blocks = (current_batch_size * CONV2_FILTERS * 8 * 8 + threads - 1) / threads;
                conv2d_kernel<<<blocks, threads>>>(d_pool1_out, d_conv2_out, d_conv2_w, d_conv2_b,
                                                   current_batch_size, CONV1_FILTERS, CONV2_FILTERS, 12, 12, 5);
                
                // Pool2
                blocks = (current_batch_size * CONV2_FILTERS * 4 * 4 + threads - 1) / threads;
                maxpool2d_kernel<<<blocks, threads>>>(d_conv2_out, d_pool2_out,
                                                      current_batch_size, CONV2_FILTERS, 8, 8, 2);
                
                // FC1
                blocks = (current_batch_size * FC1_SIZE + threads - 1) / threads;
                fc_forward_kernel<<<blocks, threads>>>(d_pool2_out, d_fc1_out, d_fc1_w, d_fc1_b,
                                                       current_batch_size, CONV2_FILTERS * 4 * 4, FC1_SIZE);
                relu_kernel<<<blocks, threads>>>(d_fc1_out, current_batch_size * FC1_SIZE);
                
                // FC2
                blocks = (current_batch_size * OUTPUT_SIZE + threads - 1) / threads;
                fc_forward_kernel<<<blocks, threads>>>(d_fc1_out, d_output, d_fc2_w, d_fc2_b,
                                                       current_batch_size, FC1_SIZE, OUTPUT_SIZE);
                
                // Softmax
                softmax_kernel<<<current_batch_size, OUTPUT_SIZE>>>(d_output, d_output, current_batch_size, OUTPUT_SIZE);
                
                CUDA_CHECK(cudaDeviceSynchronize());
                
                if (batch % 100 == 0) {
                    printf("  Batch %d/%d\r", batch, num_batches);
                    fflush(stdout);
                }
            }
            
            printf("\n");
        }
        
        printf("\n✅ Training completato!\n");
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_conv1_out);
        cudaFree(d_pool1_out);
        cudaFree(d_conv2_out);
        cudaFree(d_pool2_out);
        cudaFree(d_fc1_out);
        cudaFree(d_output);
        cudaFree(d_conv1_w);
        cudaFree(d_conv1_b);
        cudaFree(d_conv2_w);
        cudaFree(d_conv2_b);
        cudaFree(d_fc1_w);
        cudaFree(d_fc1_b);
        cudaFree(d_fc2_w);
        cudaFree(d_fc2_b);
        
        free(h_conv1_w);
        free(h_conv1_b);
        free(h_conv2_w);
        free(h_conv2_b);
        free(h_fc1_w);
        free(h_fc1_b);
        free(h_fc2_w);
        free(h_fc2_b);
    }
}