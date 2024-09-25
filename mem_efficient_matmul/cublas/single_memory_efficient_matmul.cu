#include <torch/extension.h>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "errors.h"

torch::Tensor memory_efficent_matmul(torch::Tensor &mat_a, torch::Tensor &mat_b, int tile_size) 
{
    std::vector<int64_t> a_size = mat_a.sizes().vec();
    std::vector<int64_t> b_size = mat_b.sizes().vec();
    std::vector<int64_t> a_stride = mat_a.strides().vec();
    std::vector<int64_t> b_stride = mat_b.strides().vec();

    TORCH_INTERNAL_ASSERT(a_size.size() == 2 && b_size.size() == 2,
                            "Only support 2-D matmul now!");
    TORCH_INTERNAL_ASSERT(a_size[1] == b_size[0],
                            "The dimensions of mat1 and mat2 do not match.");
    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == mat_b.scalar_type(), "Datatype of matrix a and matrix b should be same!");

    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == torch::ScalarType::Float, "Only support float for now.");

    TORCH_INTERNAL_ASSERT(mat_a.device().is_cpu() && mat_b.device().is_cpu(), "Input tensor should be put on host.");

    int64_t m = a_size[0];
    int64_t n = b_size[1];
    int64_t k = a_size[1];
    std::cout << m <<", "<< n <<", " <<k << std::endl;

    torch::Tensor out = at::zeros({m, n}, mat_a.options());

    void* out_ptr = out.data_ptr();
    void* a_ptr = mat_a.data_ptr();
    void* b_ptr = mat_b.data_ptr();

    //TODO padding
    size_t tile_num_m = m / tile_size;
    size_t tile_num_n = n / tile_size;
    size_t tile_num_k = k / tile_size;
    size_t pad_m = tile_num_m * tile_size;
    size_t pad_n = tile_num_n * tile_size;
    size_t pad_k = tile_num_k * tile_size;

    cudaStream_t stream_a, stream_b, stream_out2dev, stream_cublas, stream_out2host;
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
    CUDA_CHECK(cudaMallocPitch(&a_dev, &dpitch, tile_size*sizeof(float), tile_size));
    CUDA_CHECK(cudaMallocPitch(&b_dev, &dpitch, tile_size*sizeof(float), tile_size));
    CUDA_CHECK(cudaMallocPitch(&out_dev, &dpitch, tile_size*sizeof(float), tile_size));

    float alpha = 1.0;
    float beta = 1.0;

    for (int m_tile_idx=0; m_tile_idx<tile_num_m; m_tile_idx++)
    {
        for (int n_tile_idx=0; n_tile_idx<tile_num_n; n_tile_idx++)
        {
            size_t offset_out = m_tile_idx * tile_size * n + n_tile_idx*tile_size;
            for (int k_tile_idx=0; k_tile_idx<tile_num_k; k_tile_idx++)
            {
                size_t offset_a = m_tile_idx * tile_size * k + k_tile_idx * tile_size;
                size_t offset_b = k_tile_idx * tile_size * n + n_tile_idx * tile_size;
                // copy a tile of A to GPU
                CUDA_CHECK(cudaStreamWaitEvent(stream_a, event_out2host));
                CUDA_CHECK(cudaMemcpy2DAsync(a_dev, dpitch, (float*)a_ptr+offset_a, k*sizeof(float), tile_size*sizeof(float), tile_size, cudaMemcpyHostToDevice, stream_a));
                CUDA_CHECK(cudaEventRecord(event_a, stream_a));

                // copy a tile of B to GPU
                CUDA_CHECK(cudaMemcpy2DAsync(b_dev, dpitch, (float*)b_ptr+offset_b, n*sizeof(float), tile_size*sizeof(float), tile_size, cudaMemcpyHostToDevice, stream_b));
                CUDA_CHECK(cudaEventRecord(event_b, stream_b));

                // copy a tile of C to GPU
                CUDA_CHECK(cudaStreamWaitEvent(stream_out2dev, event_out2dev));
                CUDA_CHECK(cudaMemcpy2DAsync(out_dev, dpitch, (float*)out_ptr+offset_out, n*sizeof(float), tile_size*sizeof(float), tile_size, cudaMemcpyHostToDevice, stream_out2dev));
                CUDA_CHECK(cudaEventRecord(event_out2dev, stream_out2dev));

                // invoke cublasSgemm C=A@B+C
                CUDA_CHECK(cudaStreamWaitEvent(stream_cublas, event_a));
                CUDA_CHECK(cudaStreamWaitEvent(stream_cublas, event_b));
                CUDA_CHECK(cudaStreamWaitEvent(stream_cublas, event_out2dev));
                CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, tile_size, tile_size, tile_size, &alpha, (float*)b_dev, dpitch/sizeof(float), (float*)a_dev, dpitch/sizeof(float), &beta, (float*)out_dev, dpitch/sizeof(float)));
                CUDA_CHECK(cudaEventRecord(event_cublas, stream_cublas));

                // copy C to CPU
                CUDA_CHECK(cudaStreamWaitEvent(stream_out2host, event_cublas));
                CUDA_CHECK(cudaMemcpy2DAsync((float*)out_ptr+offset_out, n*sizeof(float), out_dev, dpitch, tile_size*sizeof(float), tile_size, cudaMemcpyDeviceToHost, stream_out2host));
                CUDA_CHECK(cudaEventRecord(event_out2host, stream_out2host));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm", &memory_efficent_matmul,
        "Fused kernel of concat and embedding");
}
