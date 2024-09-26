// do-while is used to create a local area to avoid duplicated variable names.
#define CUDA_CHECK(EXPRESS)                                                   \
    do {                                                                      \
        cudaError_t error = EXPRESS;                                          \
        if (error != cudaSuccess) {                                           \
            std::cerr << "The CUDA api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::cerr << cudaGetErrorName(error) << std::endl;                \
            std::cerr << cudaGetErrorString(error)<< std::endl;                \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(EXPRESS)                                                 \
    do {                                                                      \
        cublasStatus_t status = EXPRESS;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "The cuBLAS api failed in file: " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::exit(-1);                                                    \
        }                                                                     \
    } while (0)