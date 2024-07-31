#include <torch/extension.h>
#include <vector>

void fused_concat_embedding_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& weight, int dim);

std::vector<int64_t> get_out_shape(std::vector<torch::Tensor>& inputs, torch::Tensor& weight, int dim){
    int dim_sum = 0;
    int num_input = inputs.size();
    int ndim = inputs[0].dim();
    std::vector<int64_t> out_shape;

    for (int i=0; i<num_input; i++){
        dim_sum += inputs[i].size(dim);
    }

    for (int i=0; i<ndim; i++){
        if (i != dim)
        {
            out_shape.push_back(inputs[i].size(i));
        }
        else
        {
            out_shape.push_back(dim_sum);
        }
    }
    out_shape.push_back(weight.size(1));
    return out_shape;
}

int get_unwrapper_dim(int dim, int ndim){
    if (dim >= 0)
    {
        return dim;
    }
    else
    {
        return ndim - std::abs(dim);
    }
}

torch::Tensor fused_cat_embedding(std::vector<torch::Tensor>& inputs, torch::Tensor& weight, int dim){
    // compute the shape of output, create out tensor

    int num_input = inputs.size();
    int ndim = inputs[0].dim();
    TORCH_INTERNAL_ASSERT(std::abs(dim) < ndim, "dim shoud not be greater than the dim of input tensor.");
    TORCH_INTERNAL_ASSERT(weight.dim()==2, "weight must be 2-D tensor.");
    TORCH_INTERNAL_ASSERT(ndim<=4, "for better performance, fused concat-embedding op only support dimension <= 4.");
    dim = get_unwrapper_dim(dim, ndim);
    bool contiguous = true;
    bool same_dim = true;
    for (auto input: inputs)
    {
        contiguous = input.is_contiguous() && contiguous;
        if (input.dim() != ndim)
        {
            same_dim = false;
        }
    }
    TORCH_INTERNAL_ASSERT(contiguous == true, "fused cat-embedding op only support contiguous inputs now.");
    TORCH_INTERNAL_ASSERT(same_dim == true, "input tensors must have same dimension,");

    bool illegal_shape = false;
    for (int i=0; i<ndim; i++)
    {
        if (i == dim) continue;
        for (int j=1; j<num_input; j++)
        {  
           if (inputs[j].size(i) != inputs[j-1].size(i))
           {
                illegal_shape = true;
           } 
        }
    }
    TORCH_INTERNAL_ASSERT(!illegal_shape, "the shape of input tensors must be same except for the concat dim");
      
    std::vector<int64_t> out_shape = get_out_shape(inputs, weight, dim);
    torch::Tensor out = torch::empty(out_shape, inputs[0].options().dtype(at::ScalarType::Float));

    fused_concat_embedding_cuda(out, inputs, weight, dim);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_cat_embedding", &fused_cat_embedding, "Fused kernel of concat and embedding");
}



