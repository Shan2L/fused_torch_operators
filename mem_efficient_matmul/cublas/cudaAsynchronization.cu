#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>

// do-while is used to create a local area to avoid duplicated variable names.
#define CUDA_CHECK(EXPRESS)                                                   \
    do {                                                                      \
        cudaError_t error = EXPRESS;                                          \
        if (error != cudaSuccess) {                                           \
            std::cerr << "The cublas api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::cerr << cudaGetErrorName(error) << std::endl;                \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

__global__ void print_device(float* data, int num) {
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x < num) {
        printf("data: %f \n", data[global_x]);
    }
}

int main() {
    std::vector<float> in1(100);
    std::vector<float> in2(100);
    std::vector<float> in3(100);

    for (int i = 0; i < 100; i++) {
        in1[i] = 1;
        in2[i] = 2;
        in3[i] = 3;
    }

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));

    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    void *in1_dev, *in2_dev, *in3_dev;

    CUDA_CHECK(
        cudaMallocAsync(&in1_dev, in1.size() * sizeof(float), stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(in1_dev, in1.data(), in1.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream[0]));
    print_device<<<1, 128, 0, stream[0]>>>((float*)in1_dev, 100);

    CUDA_CHECK(
        cudaMallocAsync(&in2_dev, in2.size() * sizeof(float), stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(in2_dev, in2.data(), in2.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream[0]));
    print_device<<<1, 128, 0, stream[0]>>>((float*)in2_dev, 100);
    CUDA_CHECK(cudaEventRecord(event, stream[0]));

    CUDA_CHECK(cudaStreamWaitEvent(stream[1], event));
    CUDA_CHECK(
        cudaMallocAsync(&in3_dev, in3.size() * sizeof(float), stream[1]));
    CUDA_CHECK(cudaMemcpyAsync(in3_dev, in3.data(), in3.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream[1]));
    print_device<<<1, 128, 0, stream[1]>>>((float*)in3_dev, 100);

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}