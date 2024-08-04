#include <torch/extension.h>
#include <vector>

void fused_stack_add_index_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& add, torch::Tensor& index, int64_t stack_dim, int64_t index_dim);

std::vector<int64_t> get_out_shape(std::vector<torch::Tensor>& inputs, torch::Tensor& index, int64_t stack_dim, int64_t index_dim){
    int64_t input_ndim = inputs[0].dim();
    int64_t index_ndim = index.dim();
    int64_t num_input = inputs.size();
    std::vector<int64_t> out_shape;
    for (int64_t i=0; i<input_ndim; i++)
    {
        out_shape.emplace_back(inputs[0].size(i));
    }
    out_shape.insert(out_shape.begin()+stack_dim, num_input);
    out_shape.erase(out_shape.begin()+index_dim);
    for (int64_t i=index_ndim-1; i>=0; i--)
    {
        out_shape.insert(out_shape.begin()+index_dim, index.size(i));
    }

    return out_shape;
}

int64_t get_unwrapper_dim(int64_t dim, int64_t ndim){
    return dim >=0 ? dim : (ndim - std::abs(dim));
}

torch::Tensor fused_stack_add_index(std::vector<torch::Tensor>& inputs, torch::Tensor& add, torch::Tensor& index, int64_t stack_dim, int64_t index_dim){
    int64_t num_input = inputs.size();
    int64_t input_ndim = inputs[0].dim();
    int64_t index_ndim = index.dim();
    TORCH_INTERNAL_ASSERT(std::abs(stack_dim) < input_ndim, "dim of stack operation shoud not be greater than the dimension of input tensor.");
    TORCH_INTERNAL_ASSERT(input_ndim<=4, "for better performance, fused stack-add-index op only support input dimension <= 4.");
    TORCH_INTERNAL_ASSERT(index_ndim<=3, "for better performance, fused stack-add-index op inly support index dimension <= 3.");
    int64_t unwrapper_stack_dim = get_unwrapper_dim(stack_dim, input_ndim);

    bool same_dim = true;
    for (auto input: inputs)
    {
        if (input.dim() != input_ndim) same_dim = false;
    }

    TORCH_INTERNAL_ASSERT(same_dim == true, "input tensors must have same dimension,");
    bool same_dtype = true;
    bool illegal_shape = false;
    for (int64_t i=0; i<input_ndim; i++)
    {
        if (inputs[i].scalar_type() != inputs[0].scalar_type()) same_dtype = false;
        if (i == unwrapper_stack_dim) continue;
        for (int64_t j=1; j<num_input; j++)
        {  
           if (inputs[j].size(i) != inputs[j-1].size(i)) illegal_shape = true;
        }
    }
    TORCH_INTERNAL_ASSERT(!illegal_shape, "the shape of input tensors must be same except for the concat dim");
    TORCH_INTERNAL_ASSERT(same_dtype, "Input tensors should have same data type.");
    TORCH_INTERNAL_ASSERT((inputs[0].scalar_type() == torch::ScalarType::Float) || (inputs[0].scalar_type() == torch::ScalarType::Half), "Only support input tensor with datatype float or half.");
    TORCH_INTERNAL_ASSERT((index.scalar_type() == torch::ScalarType::Int) || (index.scalar_type() == torch::ScalarType::Long), "Only support index tensor with datatype int or long.");

    std::vector<int64_t> out_shape = get_out_shape(inputs, index, unwrapper_stack_dim, index_dim);
    torch::Tensor out = torch::empty(out_shape, inputs[0].options().dtype(inputs[0].scalar_type()));
    fused_stack_add_index_cuda(out, inputs, add, index, unwrapper_stack_dim, index_dim);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_stack_add_index, "Fused kernel of concat and embedding");
}



