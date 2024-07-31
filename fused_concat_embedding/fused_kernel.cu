#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

template <typename scalar_t>
void debug_num(std::string str, scalar_t val)
{
    std::cout << val <<std::endl;
}

template <typename scalar_t>
void debug_vector(std::string str, std::vector<scalar_t>& vec)
{
    std::cout << str << ": ";
    for (auto item: vec)
    {
        std::cout << item << ", "; 
    }
    std::cout << std::endl;
}

void debug_point(std::string str)
{
    std::cout << str << std::endl;
}

std::vector<int> get_stride_by_shape(const std::vector<int>& shape)
{
    std::vector<int> stride;
    for (auto iter=shape.begin()+1; iter!=shape.end(); iter++)
    {
        stride.push_back(std::accumulate(iter, shape.end(), 1, std::multiplies<int>()));
    }
    stride.push_back(1);    
    return stride;
}

std::vector<int> get_cat_out_shape(const std::vector<torch::Tensor>& inputs, int dim){
   int dim_sum = 0;
    for (auto input: inputs)
    {
        dim_sum += input.size(dim);
    }

   int ndim = inputs[0].dim();
    std::vector<int> out_shape;
    for (int i=0; i<ndim; i++)
    {
        if (i != dim)
        {
            out_shape.push_back(inputs[i].size(i));
        }
        else
        {
            out_shape.push_back(dim_sum);
        }
    }
    return out_shape;
}

template <int ndim>
struct CatMetadata
{
public:
    int numel;
    int ndim_ = ndim;
    int dim;
    int stride[ndim];
    int size[ndim];
};

template <int ndim>
struct InputMetadata
{
public:
    int ndim_ = ndim;
    int* data_ptr ;
    int size[ndim];
    int stride[ndim];
};

template <int ndim>
struct Coord
{
public:
    Coord()
    {
        for (int i=0; i<ndim; i++)
        {
            index[i] = -1;
        }
    }
    int ndim_ = ndim;
    int index[ndim];
};


template <int ndim>
__global__ 
void fused_kernel(
        float* out,
        CatMetadata<ndim> cat_metadata,
        int num_input,
        InputMetadata<ndim>* input_metadatas,
        Coord<ndim> cat_coord,
        Coord<ndim> src_coord,
        float* weight_ptr,
        int weight_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < cat_metadata.numel) {
        int remain_index = idx;
        for (int i=0; i<ndim; i++)
        {
            int current_dim_index = remain_index/cat_metadata.stride[i];
            cat_coord.index[i] = current_dim_index;
            remain_index -=  current_dim_index * cat_metadata.stride[i];
        }

        int sum = 0, tensor_index = 0, tensor_inner_index = 0;
        for (int i=0; i<num_input; i++){
            if (sum + input_metadatas[i].size[cat_metadata.dim] > cat_coord.index[cat_metadata.dim]){
                tensor_index = i;
                tensor_inner_index = cat_coord.index[cat_metadata.dim] - sum;
                break;
            }
            else
            {
                sum += input_metadatas[i].size[cat_metadata.dim];
            }
        }

        for (int i=0; i<cat_metadata.ndim_; i++)
        {
            if (i==cat_metadata.dim)
            {
                src_coord.index[i] = tensor_inner_index;
            }
            else
            {
                src_coord.index[i] = cat_coord.index[i];
            }
        }

        int src_index =0;
        for (int i=0; i<cat_metadata.ndim_; i++)
        {
            src_index += src_coord.index[i] * input_metadatas[tensor_index].stride[i];
        }

        int ele_val = input_metadatas[tensor_index].data_ptr[src_index];
        float* src_ptr = weight_ptr + ele_val * weight_width;
        float* dst_ptr = out + (idx * weight_width);
        for (int i=0; i<weight_width; i++)
        {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

template<int ndim>
void launch_kernel(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& weight, int dim)
{
    std::vector<int> cat_shape = get_cat_out_shape(inputs, dim);
    std::vector<int> cat_stride = get_stride_by_shape(cat_shape);
    int cat_numel = std::accumulate(cat_shape.begin(), cat_shape.end(), 1, std::multiplies<int>());
    int num_input = inputs.size();

    struct CatMetadata<ndim> cat_metadata;
    cat_metadata.numel = cat_numel;
    cat_metadata.dim = dim;
    for (int i=0; i<ndim; i++)
    {
        cat_metadata.stride[i] = cat_stride[i];
        cat_metadata.size[i] = cat_shape[i];
    }

    std::vector<InputMetadata<ndim>> input_metadatas;
    for(int i=0; i<num_input; i++)
    {  
        struct InputMetadata<ndim> input_meta;
        for (int j=0; j<ndim; j++)
        {
            input_meta.stride[j] = inputs[i].stride(j);
            input_meta.size[j] = inputs[i].size(j);
        } 
        input_meta.data_ptr = inputs[i].data_ptr<int>();
        input_metadatas.push_back(input_meta);
    }

    InputMetadata<ndim>* in_meta_d;
    cudaMalloc(&in_meta_d, num_input * sizeof(InputMetadata<3>));
    cudaMemcpy(in_meta_d, input_metadatas.data(), num_input * sizeof(InputMetadata<3>), cudaMemcpyHostToDevice);
    
    struct Coord<ndim> cat_coord;
    struct Coord<ndim> src_coord;

    int block = 512;
    int grid = (cat_numel + block -1) /block;

    fused_kernel<ndim><<<grid, block>>>(
        out.data_ptr<float>(),
        cat_metadata,
        num_input,
        in_meta_d,
        cat_coord,
        src_coord,
        weight.data_ptr<float>(),
        weight.size(1)
   );
    cudaDeviceSynchronize();
    cudaFree(in_meta_d);
}


void fused_concat_embedding_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& weight, int dim){
    int ndim = inputs[0].dim();
  
    if (ndim == 1) 
    {
        launch_kernel<1>(out, inputs, weight, dim);
    }
    else if (ndim == 2)
    {
        launch_kernel<2>(out, inputs, weight, dim);
    }
    else if (ndim == 3)
    {
        launch_kernel<3>(out, inputs, weight, dim);
    }
    else /* ndim == 4 */
    {
        launch_kernel<4>(out, inputs, weight, dim);
    }
}



