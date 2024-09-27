


torch::Tensor cublas_meffi_mm_bca(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size)
{

    std::vector<int64_t> a_size = 
}


torch::Tensor cublas_meffi_mm_bcb(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size)
{

}




torch::Tensor cublas_meffi_mm_wbc(torch::Tensor& mat_a, torch::Tensor& mat_b, int tile_size, bool broadcast_a)
{
    if (broadcast_a)
    {
        return cublas_meffi_mm_bca(mat_a, mat_b, tile_size);
    }
    else
    {
        return cublas_meffi_mm_bcb(mat_a, mat_b, tile_size);
    }
}