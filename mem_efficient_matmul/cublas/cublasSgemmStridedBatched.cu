#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
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

#define CUBLAS_CHECK(EXPRESS)                                                 \
    do {                                                                      \
        cublasStatus_t status = EXPRESS;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "The cublas api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

std::vector<float> BatchedCpuMatmul(const std::vector<float>& in_a,
                                    const std::vector<float>& in_b, int batch,
                                    int m, int n, int k) {
    std::vector<float> out(batch * m * n, 0);
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    out[b * m * n + i * n + j] += in_a[b * m * k + i * k + l] *
                                                  in_b[b * k * n + l * n + j];
                }
            }
        }
    }

    return out;
}

int main() {
    int batch = 8;
    int m = 16;
    int n = 64;
    int k = 1024;
    std::vector<float> input_a(batch * m * k);
    std::vector<float> input_b(batch * k * n);
    std::vector<float> out_c(batch * m * n, 0);

    // Initializing host data
    for (int i = 0; i < batch * m * k; i++) {
        input_a.emplace_back(rand() % 100 / 1000.);
    }
    for (int i = 0; i < batch * k * n; i++) {
        input_b.emplace_back(rand() % 100 / 2024.);
    }

    // Initializing device data
    void *a_dev, *b_dev, *c_dev;
    CUDA_CHECK(cudaMalloc(&a_dev, batch * m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_dev, batch * k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_dev, batch * m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a_dev, input_a.data(), batch * m * k * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev, input_b.data(), batch * k * n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_dev, out_c.data(), batch * m * n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Get Cpu result
    auto cpu_result = BatchedCpuMatmul(input_a, input_b, batch, m, n, k);

    // Invoke cublasSgemmBatched API
    float alpha = 1.;
    float beta = 1.;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_c = m * n;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (float*)b_dev, n,
        stride_b, (float*)a_dev, k, stride_a, &beta, (float*)c_dev, n, stride_c,
        batch));

    // Check result
    std::vector<float> out_host(batch * m * n, 0);
    CUDA_CHECK(cudaMemcpy(out_host.data(), c_dev, batch * m * n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch * m * n; i++) {
        if (std::abs(cpu_result[i] - out_host[i]) > 1e-6) {
            std::cout << "The difference is too big." << std::endl;
            std::exit(-1);
        }
    }

    return 0;
}