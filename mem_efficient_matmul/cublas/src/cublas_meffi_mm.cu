#include <torch/extension.h>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "errors.h"

torch::Tensor memory_efficent_matmul(torch::Tensor& mat_a, torch::Tensor& mat_b,
                                     int tile_size) {
    std::vector<int64_t> a_size = mat_a.sizes().vec();
    std::vector<int64_t> b_size = mat_b.sizes().vec();
    std::vector<int64_t> a_stride = mat_a.strides().vec();
    std::vector<int64_t> b_stride = mat_b.strides().vec();

    TORCH_INTERNAL_ASSERT(a_size.size() == 2 && b_size.size() == 2,
                          "Only support 2-D matmul now!");
    TORCH_INTERNAL_ASSERT(a_size[1] == b_size[0],
                          "The dimensions of mat1 and mat2 do not match.");
    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == mat_b.scalar_type(),
                          "Datatype of matrix a and matrix b should be same!");

    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == torch::ScalarType::Float,
                          "Only support float for now.");

    TORCH_INTERNAL_ASSERT(mat_a.device().is_cpu() && mat_b.device().is_cpu(),
                          "Input tensor should be put on host.");

    int64_t m = a_size[0];
    int64_t n = b_size[1];
    int64_t k = a_size[1];

    torch::Tensor out = at::zeros({m, n}, mat_a.options());

    void* out_ptr = out.data_ptr();
    void* a_ptr = mat_a.data_ptr();
    void* b_ptr = mat_b.data_ptr();

    size_t tile_num_m = (m + tile_size - 1) / tile_size;
    size_t tile_num_n = (n + tile_size - 1) / tile_size;
    size_t tile_num_k = (k + tile_size - 1) / tile_size;
    size_t pad_m = tile_num_m * tile_size;
    size_t pad_n = tile_num_n * tile_size;
    size_t pad_k = tile_num_k * tile_size;

    // TODO Remove this host malloc, the  malloc and copy operation is expensive, which downgrade the latency.
    // TODO An substitute sulotion is to pad the matrix in tile.
    float* out_ptr_pad = (float*)malloc(pad_m * pad_n * sizeof(float));
    if (out_ptr_pad == nullptr) {
        std::cout << "Failed to allocate memory." << std::endl;
        std::exit(-1);
    }
    float* a_ptr_pad = (float*)malloc(pad_m * pad_k * sizeof(float));
    if (a_ptr_pad == nullptr) {
        std::cout << "Failed to allocate memory." << std::endl;
        std::exit(-1);
    }
    float* b_ptr_pad = (float*)malloc(pad_k * pad_n * sizeof(float));
    if (b_ptr_pad == nullptr) {
        std::cout << "Failed to allocate memory." << std::endl;
        std::exit(-1);
    }
    std::fill(out_ptr_pad, out_ptr_pad + pad_m * pad_n, 0.);
    std::fill(a_ptr_pad, a_ptr_pad + pad_m * pad_k, 0.);
    std::fill(b_ptr_pad, b_ptr_pad + pad_k * pad_n, 0.);

    CUDA_CHECK(cudaMemcpy2D(out_ptr_pad, pad_n * sizeof(float), out_ptr,
                            n * sizeof(float), n * sizeof(float), m,
                            cudaMemcpyHostToHost));
    CUDA_CHECK(cudaMemcpy2D(a_ptr_pad, pad_k * sizeof(float), a_ptr,
                            k * sizeof(float), k * sizeof(float), m,
                            cudaMemcpyHostToHost));
    CUDA_CHECK(cudaMemcpy2D(b_ptr_pad, pad_n * sizeof(float), b_ptr,
                            n * sizeof(float), n * sizeof(float), k,
                            cudaMemcpyHostToHost));

    size_t hpitch_out = pad_n * sizeof(float);
    size_t hpitch_a = pad_k * sizeof(float);
    size_t hpitch_b = pad_n * sizeof(float);

    cudaStream_t stream_a, stream_b, stream_out2dev, stream_cublas,
        stream_out2host;
    cudaEvent_t event_a, event_b, event_out2dev, event_cublas, event_out2host;
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    CUDA_CHECK(cudaStreamCreate(&stream_b));
    CUDA_CHECK(cudaStreamCreate(&stream_out2dev));
    CUDA_CHECK(cudaStreamCreate(&stream_cublas));
    CUDA_CHECK(cudaStreamCreate(&stream_out2host));
    CUDA_CHECK(cudaEventCreate(&event_a));
    CUDA_CHECK(cudaEventCreate(&event_b));
    CUDA_CHECK(cudaEventCreate(&event_out2dev));
    CUDA_CHECK(cudaEventCreate(&event_cublas));
    CUDA_CHECK(cudaEventCreate(&event_out2host));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream_cublas));

    void *a_dev, *b_dev, *out_dev;
    size_t dpitch;
    CUDA_CHECK(
        cudaMallocPitch(&a_dev, &dpitch, tile_size * sizeof(float), tile_size));
    CUDA_CHECK(
        cudaMallocPitch(&b_dev, &dpitch, tile_size * sizeof(float), tile_size));
    CUDA_CHECK(cudaMallocPitch(&out_dev, &dpitch, tile_size * sizeof(float),
                               tile_size));

    float alpha = 1.0;
    float beta = 1.0;

    // TODO the asynchonization has not been enable correctly, the performance looks the same as the serial execution.
    for (int m_tile_idx = 0; m_tile_idx < tile_num_m; m_tile_idx++) {
        for (int n_tile_idx = 0; n_tile_idx < tile_num_n; n_tile_idx++) {
            size_t offset_out =
                m_tile_idx * tile_size * pad_n + n_tile_idx * tile_size;
            for (int k_tile_idx = 0; k_tile_idx < tile_num_k; k_tile_idx++) {

                size_t offset_a =
                    m_tile_idx * tile_size * pad_k + k_tile_idx * tile_size;
                size_t offset_b =
                    k_tile_idx * tile_size * pad_n + n_tile_idx * tile_size;
                // copy a tile of A to GPU
                CUDA_CHECK(cudaStreamWaitEvent(stream_a, event_out2host));
                CUDA_CHECK(cudaMemcpy2DAsync(
                    a_dev, dpitch, (float*)a_ptr_pad + offset_a, hpitch_a,
                    tile_size * sizeof(float), tile_size,
                    cudaMemcpyHostToDevice, stream_a));
                CUDA_CHECK(cudaEventRecord(event_a, stream_a));

                // copy a tile of B to GPU
                CUDA_CHECK(cudaMemcpy2DAsync(
                    b_dev, dpitch, (float*)b_ptr_pad + offset_b, hpitch_b,
                    tile_size * sizeof(float), tile_size,
                    cudaMemcpyHostToDevice, stream_b));
                CUDA_CHECK(cudaEventRecord(event_b, stream_b));

                // copy a tile of C to GPU
                CUDA_CHECK(cudaStreamWaitEvent(stream_out2dev, event_out2dev));
                CUDA_CHECK(cudaMemcpy2DAsync(
                    out_dev, dpitch, (float*)out_ptr_pad + offset_out,
                    hpitch_out, tile_size * sizeof(float), tile_size,
                    cudaMemcpyHostToDevice, stream_out2dev));
                CUDA_CHECK(cudaEventRecord(event_out2dev, stream_out2dev));

                // invoke cublasSgemm C=A@B+C
                CUDA_CHECK(cudaStreamWaitEvent(stream_cublas, event_a));
                CUDA_CHECK(cudaStreamWaitEvent(stream_cublas, event_b));
                CUDA_CHECK(cudaStreamWaitEvent(stream_cublas, event_out2dev));
                CUBLAS_CHECK(
                    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                tile_size, tile_size, tile_size, &alpha,
                                (float*)b_dev, dpitch / sizeof(float),
                                (float*)a_dev, dpitch / sizeof(float), &beta,
                                (float*)out_dev, dpitch / sizeof(float)));
                CUDA_CHECK(cudaEventRecord(event_cublas, stream_cublas));

                // copy C to CPU
                CUDA_CHECK(cudaStreamWaitEvent(stream_out2host, event_cublas));
                CUDA_CHECK(cudaMemcpy2DAsync(
                    (float*)out_ptr_pad + offset_out, hpitch_out, out_dev,
                    dpitch, tile_size * sizeof(float), tile_size,
                    cudaMemcpyDeviceToHost, stream_out2host));
                CUDA_CHECK(cudaEventRecord(event_out2host, stream_out2host));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(out_ptr, n * sizeof(float), out_ptr_pad,
                            pad_n * sizeof(float), n * sizeof(float), m,
                            cudaMemcpyHostToHost));

    CUDA_CHECK(cudaFree(a_dev));
    CUDA_CHECK(cudaFree(b_dev));
    CUDA_CHECK(cudaFree(out_dev));

    delete (a_ptr_pad);
    delete (b_ptr_pad);
    delete (out_ptr_pad);
    a_ptr_pad = nullptr;
    b_ptr_pad = nullptr;
    out_ptr_pad = nullptr;

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &memory_efficent_matmul,
          "Fused kernel of concat and embedding");
}
