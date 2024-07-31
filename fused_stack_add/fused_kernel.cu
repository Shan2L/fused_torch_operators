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

std::vector<int64_t> get_stride_by_shape(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> stride;
    for (auto iter=shape.begin()+1; iter!=shape.end(); iter++)
    {
        stride.push_back(std::accumulate(iter, shape.end(), 1, std::multiplies<int>()));
    }
    stride.push_back(1);    
    return stride;
}

std::vector<int64_t> get_stack_out_shape(const std::vector<torch::Tensor>& inputs, int dim){
    int num_input = inputs.size();
    int ndim = inputs[0].dim();
    std::vector<int64_t> out_shape;
    for (int i=0; i<ndim; i++)
    {
        out_shape.push_back(inputs[i].size(i));
    }
    out_shape.insert(out_shape.begin()+dim, num_input);
    return out_shape;
}

template <int ndim>
struct TensorMetadata
{
public:
    int ndim_ = ndim;
    float* data_ptr ;
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
        TensorMetadata<ndim+1> out_metadata,
        int64_t num_input,
        TensorMetadata<ndim>* input_metadatas,
        Coord<ndim+1> out_coord,
        Coord<ndim> src_coord,
        Coord<3> other_coord,
        TensorMetadata<3> other_metadata,
        int64_t dim,
        int64_t numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < numel) {
        int remain_index = idx;
        for (int i=0; i<ndim+1; i++)
        {
            int current_dim_index = remain_index/out_metadata.stride[i];
            out_coord.index[i] = current_dim_index;
            remain_index -=  current_dim_index * out_metadata.stride[i];
        }

        int tensor_index = out_coord.index[dim];
        size_t iter = 0;
        for (int i=0; i<out_metadata.ndim_; i++)
        {
            if (i!=dim)
            {
                src_coord.index[iter] = out_coord.index[i];
                iter++;
            }
        }

        int src_index =0;
        for (int i=0; i<input_metadatas[tensor_index].ndim_; i++)
        {
            src_index += src_coord.index[i] * input_metadatas[tensor_index].stride[i];
        }

        other_coord.index[0] = 0;
        other_coord.index[1] = out_coord.index[1];
        other_coord.index[2] = out_coord.index[3];

        int other_index = 0;
        for (int i=0; i<3; i++)
        {
            other_index += other_coord.index[i] * other_metadata.stride[i];
        }

        float ele_src = input_metadatas[tensor_index].data_ptr[src_index];
        float ele_other = other_metadata.data_ptr[other_index];
        float* dst_ptr = out_metadata.data_ptr + idx;

        *dst_ptr = ele_src + ele_other;

    }
}

template<int ndim>
void launch_kernel(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& other, int64_t dim)
{
    std::vector<int64_t> cat_shape = get_stack_out_shape(inputs, dim);
    std::vector<int64_t> cat_stride = get_stride_by_shape(cat_shape);
    int64_t numel = out.numel();
    int64_t num_input = inputs.size();

    struct TensorMetadata<ndim+1> out_metadata;
    out_metadata.data_ptr = out.data_ptr<float>();
    
    for (int i=0; i<ndim+1; i++)
    {
        out_metadata.stride[i] = cat_stride[i];
        out_metadata.size[i] = cat_shape[i];
    }
    

    std::vector<TensorMetadata<ndim>> input_metadatas;
    for(int i=0; i<num_input; i++)
    {  
        struct TensorMetadata<ndim> input_meta;
        for (int j=0; j<ndim; j++)
        {
            input_meta.stride[j] = inputs[i].stride(j);
            input_meta.size[j] = inputs[i].size(j);
        } 
        input_meta.data_ptr = inputs[i].data_ptr<float>();
        input_metadatas.push_back(input_meta);
    }

    TensorMetadata<ndim>* in_meta_d;
    cudaMalloc(&in_meta_d, num_input * sizeof(TensorMetadata<ndim>));
    cudaMemcpy(in_meta_d, input_metadatas.data(), num_input * sizeof(TensorMetadata<ndim>), cudaMemcpyHostToDevice);
    

    TensorMetadata<3> other_metadata;
    other_metadata.data_ptr = other.data_ptr<float>();
    for (size_t i=0; i<3; i++)
    {
        other_metadata.size[i] = other.size(i);
        other_metadata.stride[i] = other.stride(i);
    }


    struct Coord<ndim+1> out_coord;
    struct Coord<ndim> src_coord;
    struct Coord<3> other_coord;

    int block = 512;
    int grid = (numel + block -1) /block;

    fused_kernel<ndim><<<grid, block>>>(
        out_metadata,
        num_input,
        in_meta_d,
        out_coord,
        src_coord,
        other_coord,
        other_metadata,
        dim,
        numel
        );
    cudaDeviceSynchronize();
    cudaFree(in_meta_d);
}


void fused_stack_add_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& other, int64_t dim){
    int ndim = inputs[0].dim();
  
    if (ndim == 1) 
    {
        launch_kernel<1>(out, inputs, other, dim);
    }
    else if (ndim == 2)
    {
        launch_kernel<2>(out, inputs, other, dim);
    }
    else if (ndim == 3)
    {
        launch_kernel<3>(out, inputs, other, dim);
    }
    else /* ndim == 4 */
    {
        launch_kernel<4>(out, inputs, other, dim);
    }
}



