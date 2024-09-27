#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "errors.h"

__global__ void check_dev_data(const int id, float* data, int pitch, int sizex)
{
    printf("id: %d, batch: %d, y: %d, x: %d, data: %f\n", 
    id, blockIdx.x, threadIdx.y, threadIdx.x, 
    data[(blockIdx.x * blockDim.y + threadIdx.y) * pitch/sizeof(float) + threadIdx.x]);
}

#define debug_name(name, var) \
    std::cout << name <<": " << var <<std::endl;

torch::Tensor cublas_meffi_bmm(torch::Tensor& mat_a, torch::Tensor& mat_b,
                                     int tile_size) {
    std::vector<int64_t> a_size = mat_a.sizes().vec();
    std::vector<int64_t> b_size = mat_b.sizes().vec();
    std::vector<int64_t> a_stride = mat_a.strides().vec();
    std::vector<int64_t> b_stride = mat_b.strides().vec();
    TORCH_INTERNAL_ASSERT(a_size.size() == b_size.size(), "Dimension of two input must be same now.");
    int ndim = a_size.size();

    // TODO dimension check and broadcast

    TORCH_INTERNAL_ASSERT(a_size[ndim-1] == b_size[ndim-2],
                          "The dimensions of mat1 and mat2 do not match.");
    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == mat_b.scalar_type(),
                          "Datatype of matrix a and matrix b should be same!");




    // int64_t batch_size = std::accumulate(a_size.rbegin()+2, a_size.rend(), 1, std::multiplies<int64_t>());
    int64_t batch_size = a_size[0];
    int64_t m = a_size[ndim-2];
    int64_t n = b_size[ndim-1];
    int64_t k = a_size[ndim-1];

    debug_name("batch_size", batch_size)
    debug_name("m", m)
    debug_name("n", n)
    debug_name("k", k)


    torch::Tensor out = at::zeros({batch_size, m, n}, mat_a.options());

    void* out_ptr = out.data_ptr();
    void* a_ptr = mat_a.data_ptr();
    void* b_ptr = mat_b.data_ptr();

    // TODO Asynchronization implementation.
    // TODO replace the resource creation with pytorch builtin function.

    // cudaStream_t stream_a, stream_b, stream_out2dev, stream_cublas,
    //     stream_out2host;
    // cudaEvent_t event_a, event_b, event_out2dev, event_cublas, event_out2host;
    // CUDA_CHECK(cudaStreamCreate(&stream_a));
    // CUDA_CHECK(cudaStreamCreate(&stream_b));
    // CUDA_CHECK(cudaStreamCreate(&stream_out2dev));
    // CUDA_CHECK(cudaStreamCreate(&stream_cublas));
    // CUDA_CHECK(cudaStreamCreate(&stream_out2host));
    // CUDA_CHECK(cudaEventCreate(&event_a));
    // CUDA_CHECK(cudaEventCreate(&event_b));
    // CUDA_CHECK(cudaEventCreate(&event_out2dev));
    // CUDA_CHECK(cudaEventCreate(&event_cublas));
    // CUDA_CHECK(cudaEventCreate(&event_out2host));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    // CUBLAS_CHECK(cublasSetStream(cublas_handle, stream_cublas));

    // TODO replace the cudaMalloc with pytorch inplace function.
    // 目标内存指针设置
    cudaMemcpy3DParms copyParams_a = {0};
    cudaPitchedPtr dstPtr_a;
    CUDA_CHECK(cudaMalloc3D(&dstPtr_a, make_cudaExtent(tile_size*sizeof(float), tile_size, batch_size)));
    cudaPitchedPtr srcPtr_a = make_cudaPitchedPtr((void*)a_ptr, k*sizeof(float), k, m);
    copyParams_a.srcPtr = srcPtr_a;
    copyParams_a.dstPtr = dstPtr_a;
    copyParams_a.kind = cudaMemcpyHostToDevice;
    copyParams_a.dstPos = make_cudaPos(0, 0, 0);

    debug_name("dstPtr_a.pitch", dstPtr_a.pitch)
    debug_name("dstPtr_a.xsize", dstPtr_a.xsize)
    debug_name("dstPtr_a.ysize", dstPtr_a.ysize)

    cudaMemcpy3DParms copyParams_b = {0};
    cudaPitchedPtr dstPtr_b;
    CUDA_CHECK(cudaMalloc3D(&dstPtr_b, make_cudaExtent(tile_size*sizeof(float), tile_size, batch_size)));
    cudaPitchedPtr srcPtr_b = make_cudaPitchedPtr((void*)b_ptr, n*sizeof(float), n, k);
    copyParams_b.srcPtr = srcPtr_b;
    copyParams_b.dstPtr = dstPtr_b;
    copyParams_b.kind = cudaMemcpyHostToDevice;
    copyParams_b.dstPos = make_cudaPos(0, 0, 0);


    cudaPitchedPtr dstPtr_out;
    CUDA_CHECK(cudaMalloc3D(&dstPtr_out, make_cudaExtent(tile_size*sizeof(float), tile_size, batch_size)));
    cudaPitchedPtr srcPtr_out = make_cudaPitchedPtr((void*)out_ptr, n*sizeof(float), n, m);
    
    cudaMemcpy3DParms copyParams_out2dev = {0};
    copyParams_out2dev.srcPtr = srcPtr_out;
    copyParams_out2dev.dstPtr = dstPtr_out;
    copyParams_out2dev.kind = cudaMemcpyHostToDevice;
    copyParams_out2dev.dstPos = make_cudaPos(0, 0, 0);

    cudaMemcpy3DParms copyParams_out2host = {0};
    copyParams_out2host.srcPtr = dstPtr_out;
    copyParams_out2host.dstPtr = srcPtr_out;
    copyParams_out2host.kind = cudaMemcpyDeviceToHost;
    copyParams_out2host.srcPos = make_cudaPos(0, 0, 0);

    float alpha = 1.0;
    float beta = 1.0;

    size_t tile_num_m = (m + tile_size -1) / tile_size;
    size_t tile_num_n = (n + tile_size -1) / tile_size;
    size_t tile_num_k = (k + tile_size -1) / tile_size;

    debug_name("tile_num_m", tile_num_m)
    debug_name("tile_num_n", tile_num_n)
    debug_name("tile_num_k", tile_num_k)


    // TODO the asynchonization has not been enable correctly, the performance looks the same as the serial execution.
    for (int m_tile_idx = 0; m_tile_idx < tile_num_m; m_tile_idx++) {
        for (int n_tile_idx = 0; n_tile_idx < tile_num_n; n_tile_idx++) {
            for (int k_tile_idx = 0; k_tile_idx < tile_num_k; k_tile_idx++) {
                std::cout << m_tile_idx << ", " << n_tile_idx << ", " << k_tile_idx<< std::endl;
                size_t element_m = m % tile_size == 0 ? tile_size : (m_tile_idx == tile_num_m - 1 ? (m % tile_size) : tile_size);
                size_t element_n = n % tile_size == 0 ? tile_size : (n_tile_idx == tile_num_n - 1 ? (n % tile_size) : tile_size);
                size_t element_k = k % tile_size == 0 ? tile_size : (k_tile_idx == tile_num_k - 1 ? (k % tile_size) : tile_size);

                debug_name("element_m", element_m)
                debug_name("element_n", element_n)
                debug_name("element_k", element_k)
                // copy a tile of A to GPU

                if (element_m != tile_size || element_k != tile_size)
                {
                    CUDA_CHECK(cudaMemset3D(dstPtr_a, 0, make_cudaExtent(tile_size*sizeof(float), tile_size, batch_size)));
                }
                copyParams_a.extent = make_cudaExtent(element_k*sizeof(float), element_m, batch_size);
                copyParams_a.srcPos = make_cudaPos(k_tile_idx*tile_size*sizeof(float), m_tile_idx*tile_size, 0);
                CUDA_CHECK(cudaMemcpy3D(&copyParams_a));
                cudaDeviceSynchronize();
                check_dev_data<<<batch_size, dim3(element_k, element_m)>>>(1, (float*)dstPtr_a.ptr, dstPtr_a.pitch, dstPtr_a.xsize);

                // copy a tile of B to GPU
                if (element_n != tile_size || element_k != tile_size)
                {
                    CUDA_CHECK(cudaMemset3D(dstPtr_b, 0, make_cudaExtent(tile_size*sizeof(float), tile_size, batch_size)));
                }
                copyParams_b.extent = make_cudaExtent(element_n* sizeof(float), element_k, batch_size);
                copyParams_b.srcPos = make_cudaPos(n_tile_idx*tile_size*sizeof(float), k_tile_idx*tile_size, 0);
                CUDA_CHECK(cudaMemcpy3D(&copyParams_b));
                check_dev_data<<<batch_size, dim3(element_n, element_k)>>>(2, (float*)dstPtr_b.ptr, dstPtr_b.pitch, dstPtr_a.xsize);

                // copy a tile of C to GPU
                if (element_m != tile_size || element_n != tile_size)
                {
                    CUDA_CHECK(cudaMemset3D(dstPtr_out, 0, make_cudaExtent(tile_size*sizeof(float), tile_size, batch_size)));
                }
                copyParams_out2dev.extent = make_cudaExtent(element_n* sizeof(float), element_m, batch_size);
                copyParams_out2dev.srcPos = make_cudaPos(n_tile_idx*tile_size*sizeof(float), m_tile_idx*tile_size, 0);
                CUDA_CHECK(cudaMemcpy3D(&copyParams_out2dev));
                check_dev_data<<<batch_size, dim3(element_m, element_n)>>>(4, (float*)dstPtr_out.ptr, dstPtr_out.pitch, dstPtr_out.xsize);

                // invoke cublasSgemm C=A@B+C


                CUBLAS_CHECK(
                    cublasSgemmStridedBatched(
                    cublas_handle, 
                    CUBLAS_OP_N, 
                    CUBLAS_OP_N, 
                    tile_size, tile_size, tile_size, 
                    &alpha, 
                    (float*)dstPtr_b.ptr, dstPtr_b.pitch/sizeof(float), dstPtr_b.pitch/sizeof(float)*tile_size, 
                    (float*)dstPtr_a.ptr, dstPtr_a.pitch/sizeof(float), dstPtr_a.pitch/sizeof(float)*tile_size, 
                    &beta, 
                    (float*)dstPtr_out.ptr, dstPtr_out.pitch/sizeof(float), dstPtr_out.pitch/sizeof(float)*tile_size,
                    batch_size));

                check_dev_data<<<batch_size, dim3(element_m, element_n)>>>(3, (float*)dstPtr_out.ptr, dstPtr_out.pitch, dstPtr_out.xsize);


                // CUDA_CHECK(cudaEventRecord(event_cublas, stream_cublas));
                cudaDeviceSynchronize();
                // copy C to CPU


                copyParams_out2host.dstPos = make_cudaPos(n_tile_idx*tile_size*sizeof(float), m_tile_idx*tile_size, 0);
                std::cout << "cudapos: " << "x: "<<n_tile_idx*tile_size << ", y: "<<m_tile_idx*tile_size, 
                copyParams_out2host.extent = make_cudaExtent(element_n* sizeof(float), element_m, batch_size);
                CUDA_CHECK(cudaMemcpy3D(&copyParams_out2host));

                std::cout << "output host: "<<std::endl;
                std::cout << "==============================================="<< std::endl;
                for (int b=0; b<batch_size; b++)
                {
                    for (int i=0; i<m; i++)
                    {
                        for(int j=0; j<n; j++)
                        {
                            std::cout << ((float*)out_ptr)[b*m*n+i*n+j] << ", ";
                        }   
                        std::cout << std::endl;
                    }
                    std::cout <<std::endl;
                }

                std::cout << "==============================================="<< std::endl;
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaFree(a_dev));
    // CUDA_CHECK(cudaFree(b_dev));
    // CUDA_CHECK(cudaFree(out_dev));   
    
    return out;
}

