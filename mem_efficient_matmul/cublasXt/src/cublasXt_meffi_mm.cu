#include <algorithm>
#include <cublasXt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <vector>

void CudaDeviceCheck() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error == cudaErrorNoDevice) {
        std::cerr << "No CUDA devices found." << std::endl;
        std::exit(-1);
    } else if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned error code " << error
                << ", no CUDA devices found." << std::endl;
        std::exit(-1);
    }
}


torch::Tensor memory_efficent_matmul(torch::Tensor &mat_a, torch::Tensor &mat_b, float alpha, float beta, int tile_size) 
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

    int64_t m = a_size[0];
    int64_t n = b_size[1];
    int64_t k = a_size[1];

    torch::Tensor out = at::empty({m, n}, mat_a.options());

    float* out_ptr = out.data_ptr<float>();
    float* a_ptr = mat_a.data_ptr<float>();
    float* b_ptr = mat_b.data_ptr<float>();
    
    
    if (out.scalar_type() == torch::ScalarType::Float)
    {
        CudaDeviceCheck();
        cublasStatus_t status;
        cublasXtHandle_t handle;
        status = cublasXtCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasXt handle create failed with error code:  " << status << std::endl;
            std::exit(-1);
        }
        

        int devices[1] = {0};
        status = cublasXtDeviceSelect(handle, 1, devices);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasXt device select failed with error code:  " << status << std::endl;
        }

        status = cublasXtSetBlockDim(handle, tile_size);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasXt set blockDim failed with error code:  " << status << std::endl;
            std::exit(-1);
        }
        
        status = cublasXtSgemm(handle,
                        CUBLAS_OP_N, // transa
                        CUBLAS_OP_N, // transb
                        n,           // m
                        m,           // n
                        k,           // k
                        &alpha,      // *alpha
                        b_ptr,           // *A
                        n,         // lda
                        a_ptr,           // *B
                        k,         // ldb
                        &beta,       // *beta
                        out_ptr,         // *C
                        n);        // ldc

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasXt sgemm failed with error code: " << status
                    << std::endl;
        }

        // Synchronize device
        cudaDeviceSynchronize();
        cublasXtDestroy(handle);
    }
    else 
    {
        TORCH_INTERNAL_ASSERT(false, "only support float and half datatype now.");
    }

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &memory_efficent_matmul,
        "Fused kernel of concat and embedding");
}
