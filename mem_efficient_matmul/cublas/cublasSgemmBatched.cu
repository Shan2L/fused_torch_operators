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

std::vector<float> CpuMatmul2D(const std::vector<float>& in_a,
                               const std::vector<float>& in_b, int m, int n,
                               int k) {
    std::vector<float> out(m * n, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                out[i * n + j] += in_a[i * k + l] * in_b[l * n + j];
            }
        }
    }
    return out;
}

std::vector<std::vector<float>> CpuKernel(
    const std::vector<std::vector<float>>& in_a,
    const std::vector<std::vector<float>>& in_b, int batch, int m, int n,
    int k) {
    std::vector<std::vector<float>> out;

    for (int b = 0; b < batch; b++) {
        out.emplace_back(CpuMatmul2D(in_a[b], in_b[b], m, n, k));
    }
    return out;
}

int main() {
    int batch = 8;
    int m = 512;
    int n = 128;
    int k = 1024;
    std::vector<std::vector<float>> input_a(batch);
    std::vector<std::vector<float>> input_b(batch);
    std::vector<std::vector<float>> out_c(batch, std::vector<float>(m * n, 0));
    float* a_ptrs[batch];
    float* b_ptrs[batch];
    float* c_ptrs[batch];

    // Initializing host data
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m * k; i++) {
            input_a[b].emplace_back(rand() % 100 / 1000.);
        }
        for (int i = 0; i < k * n; i++) {
            input_b[b].emplace_back(rand() % 100 / 2024.);
        }
    }

    // Initializing device data
    for (int b = 0; b < batch; b++) {
        void *a_dev, *b_dev, *c_dev;
        CUDA_CHECK(cudaMalloc(&a_dev, m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_dev, k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_dev, m * n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(a_dev, input_a[b].data(), m * k * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_dev, input_b[b].data(), k * n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c_dev, out_c[b].data(), m * n * sizeof(float),
                                cudaMemcpyHostToDevice));
        a_ptrs[b] = static_cast<float*>(a_dev);
        b_ptrs[b] = static_cast<float*>(b_dev);
        c_ptrs[b] = static_cast<float*>(c_dev);
    }

    void** a_ptr_dev;
    void** b_ptr_dev;
    void** c_ptr_dev;
    CUDA_CHECK(cudaMalloc(&a_ptr_dev, batch*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&b_ptr_dev, batch*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&c_ptr_dev, batch*sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(a_ptr_dev, a_ptrs, batch*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_ptr_dev, b_ptrs, batch*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_ptr_dev, c_ptrs, batch*sizeof(float*), cudaMemcpyHostToDevice));



    // Get Cpu result
    auto cpu_result = CpuKernel(input_a, input_b, batch, m, n, k);

    // Invoke cublasSgemmBatched API
    float alpha = 1.;
    float beta = 1.;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                    &alpha, (float**)b_ptr_dev, n, (float**)a_ptr_dev, k,
                                    &beta, (float**)c_ptr_dev, n, batch));

    // Check result
    std::vector<std::vector<float>> out_host(batch, std::vector<float>(m*n, 0));
    for (int b = 0; b < batch; b++) {
        float* c_dev = (float*)c_ptrs[b];
        CUDA_CHECK(cudaMemcpy(out_host[b].data(), c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));

    }

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m * n; i++) {
            float diff = abs(cpu_result[b][i] - out_host[b][i]);
            if (std::abs(cpu_result[b][i] - out_host[b][i]) > 1e-6)
            {
                std::cout << "The difference is too big." << std::endl;
                std::exit(-1);
            }
        }
    }

    return 0;
}