#include <torch/extension.h>

void PrepareAndLaunch(torch::Tensor& out, const torch::Tensor& in1,
                      const Tensor& in2, int64_t tile_size, int64_t m,
                      int64_t n, int64_t k) {
    TORCH_INTERNAL_ASSERT(in1.scalar_type() == torch::ScalarType::Float,
                          "Only support for float datatype now.");
    //TODO datatype dispatch here

    //TODO zero padding for input and output tensor here

    float* in1_ptr = in1.data_ptr<float>();
    float* in2_ptr = in2.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    std::vector<int64_t> stride_in1 = in1.strides().vec();
    std::vector<int64_t> stride_in2 = in2.strides().vec();
    std::vector<int64_t> stride_out = out.strides().vec();

    int64_t tilenum_m = (m + tile_size - 1) / tile_size;
    int64_t tilenum_n = (n + tile_size - 1) / tile_size;
    int64_t tilenum_k = (k + tile_size - 1) / tile_size;
    int64_t tileres_m = m % tile_size;
    int64_t tileres_n = n % tile_size;
    int64_t tileres_k = k % tile_size;

    for (int64_t tile_idx_m = 0; tile_idx_m < tilenum_m; tile_idx_m++) {
        for (int64_t tile_idx_n = 0; tile_idx_n < tilenum_n; tile_idx_n++) {
            for (int64_t tile_idx_k = 0; tile_idx_k < tilenum_k; tile_idx_k++) {

            }
        }
    }
}

torch::Tensor memory_efficient_matmul(const Tensor& in1, const Tensor& in2,
                                      const int64_t tile_size) {

    // TODO tensor shape broadcast here

    std::vector<int64_t> size_in1 = in1.sizes().vec();
    std::vector<int64_t> size_in2 = in2.sizes().vec();
    TORCH_INTERNAL_ASSERT(size_in1.size() == 3,
                          "The dimension of inpute tensor must be equal to 3");
    TORCH_INTERNAL_ASSERT(size_in2.size() == 3,
                          "The dimension of inpute tensor must be equal to 3");
    TORCH_INTERNAL_ASSERT(size_in1[2] == size_in2[1],
                          "The shape of input tensors does not match.");
    TORCH_INTERNAL_ASSERT(in1.device().is_cuda() && in2.device().is_cuda(),
                          "The input tensor should be located on device.");

    int64_t batch_size = size_in1[0];
    int64_t m = size_in1[1];
    int64_t n = size_in2[2];
    int64_t k = size_in1[2];
    std::vector<int64_t> size_out{batch_size, m, n};
    torch::Tensor out = torch::zeros(size_out, in1.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("memory_efficient_matmul", &memory_efficient_matmul,
          "Memory efficient batched matmul implementation based on cublas.");
}
