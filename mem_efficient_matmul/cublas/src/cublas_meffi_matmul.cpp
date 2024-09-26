#include <torch/extension.h>


torch::Tensor cublas_meffi_bmm(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size);

torch::Tensor cublas_meffi_mm(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm", &cublas_meffi_bmm,
          "Fused kernel of concat and embedding");
    m.def("mm", &cublas_meffi_mm,
        "Fused kernel of concat and embedding");
}