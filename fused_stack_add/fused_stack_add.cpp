#include <torch/extension.h>
#include <vector>

void fused_stack_add_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& other, int64_t dim);

std::vector<int64_t> get_out_shape(std::vector<torch::Tensor>& inputs, int64_t dim){
    int64_t num_input = inputs.size();
    int64_t ndim = inputs[0].dim();
    std::vector<int64_t> out_shape;
    for (int64_t i=0; i<ndim; i++)
    {
        out_shape.push_back(inputs[0].size(i));
    }
    out_shape.insert(out_shape.begin()+dim, num_input);
    return out_shape;
}

int64_t get_unwrapper_dim(int64_t dim, int64_t ndim){
    if (dim >= 0)
    {
        return dim;
    }
    else
    {
        return ndim - std::abs(dim);
    }
}

torch::Tensor fused_stack_add(std::vector<torch::Tensor>& inputs, torch::Tensor& other, int64_t dim){
    // compute the shape of output, create out tensor

    int64_t num_input = inputs.size();
    int64_t ndim = inputs[0].dim();
    TORCH_INTERNAL_ASSERT(std::abs(dim) < ndim, "dim shoud not be greater than the dim of input tensor.");
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
    for (int64_t i=0; i<ndim; i++)
    {
        if (i == dim) continue;
        for (int64_t j=1; j<num_input; j++)
        {  
           if (inputs[j].size(i) != inputs[j-1].size(i))
           {
                illegal_shape = true;
           } 
        }
    }
    TORCH_INTERNAL_ASSERT(!illegal_shape, "the shape of input tensors must be same except for the concat dim");
      
    std::vector<int64_t> out_shape = get_out_shape(inputs, dim);
    torch::Tensor out = torch::empty(out_shape, inputs[0].options().dtype(at::ScalarType::Float));
    fused_stack_add_cuda(out, inputs, other, dim);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_stack_add, "Fused kernel of concat and embedding");
}



