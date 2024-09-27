#include <torch/extension.h>
#include <algorithm>

torch::Tensor cublas_meffi_bmm(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size);

torch::Tensor cublas_meffi_mm(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size);

torch::Tensor cublas_meffi_mm_wbc(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size, bool broadcast_a);

torch::tensor cublas_meffi_matmul(const Tensor& mat_a, const Tensor& mat_b, int tile_size)
{
    TORCH_INTERNAL_ASSERT(mat_a.is_contiguous() && mat_b.is_contiguous(), "meffi_matmul only support contiguous memory layout.");
    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == torch::ScalarType::Float,
                          "Only support float for now.");

    TORCH_INTERNAL_ASSERT(mat_a.device().is_cpu() && mat_b.device().is_cpu(),
                          "Input tensor should be put on host.");
    std::vector<int64_t> a_size = mat_a.sizes().vec();
    std::vector<int64_t> b_size = mat_b.sizes().vec();

    TORCH_INTERNAL_ASSERT(mat_a.scalar_type() == mat_b.scalar_type(), "The datatype of two input tensor must be same.");

    if (a_size().size() == 2 && b_size.size() == 2)
    {
        TORCH_INTERNAL_ASSERT(a_size[1] == b_size[0], "The input tensor's shape are illegal.");
        return cublas_meffi_mm(mat_a, mat_b, tile_size);
    }
    else if (a_size.size() > 2 && b_size.size() > 2)
    {
        TORCH_INTERNAL_ASSERT(std::equal(s_size.beign(), a_size.end(), b_size().begin()), "If two input tensor's demension are all greater than 2, they must be same.");
        int ndim = a_size.size();
        TORCH_INTERNAL_ASSERT(a_size[ndim-1] == b_size[ndim-2], "The input tensor's shape are illegal.");
        return cublas_meffi_mm_wbc(mat_a, mat_b, tile_size);
    }
    else if (a_size.size() > 2 || b_size.size() > 2)
    {
        if(a_size.size() > 2)
        {
            TORCH_INTERNAL_ASSERT(b_size.size() == 2, "If dimension of one input is greater than 2, the other's must be equal to 2.");
            int ndim = a.size();
            TORCH_INTERNAL_ASSERT(a_size[ndim-1] == b_size[0], "The input tensor's shape are illegal.");
            return cublas_meffi_mm_wbc(mat_a, mat_b, tile_size, false);
        }
        else
        {
            TORCH_INTERNAL_ASSERT(a_size.size() == 2, "If dimension of one input is greater than 2, the other's must be equal to 2.");
            int ndim = b.size();
            TORCH_INTERNAL_ASSERT(b_size[ndim-1] == a_size[0], "The input tensor's shape are illegal.");
            return cublas_meffi_mm_wbc(mat_a, mat_b, tile_size, true);
        }

    }
    else{
        TORCH_INTERNAL_ASSERT(false, "The shape of input tensor are not supported yet.");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &cublas_meffi_matmul,
          "Fused kernel of concat and embedding");
}