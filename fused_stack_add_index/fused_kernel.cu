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
void debug_short_vector(std::string str, std::vector<scalar_t>& vec)
{
    std::cout << str << ": ";
    for (auto item: vec)
    {
        std::cout << item << ", "; 
    }
    std::cout << std::endl;
}

template<typename scalar_t>
void debug_long_vector(std::string str, std::vector<scalar_t>& vec)
{
    std::cout << str << ": " << std::endl;
    for (auto item: vec)
    {
        std::cout << item << std::endl;
    }
}

__device__ void debug_arr(int64_t* arr, int64_t size)
{
    for (int64_t i=0; i<size; i++)
    {
        printf("%d, ", (int)arr[i]);
    }
    printf("\n");
}


template <typename scalar_t>
void debug_vector(std::string str, std::vector<scalar_t>& vec)
{
    if (vec.size() >=6 )
    {
        debug_long_vector<scalar_t>(str, vec);
    }
    else
    {
        debug_short_vector<scalar_t>(str, vec);
    }
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

std::vector<int64_t> get_out_shape_by_stack_shape(const std::vector<int64_t> stack_shape, const std::vector<int64_t> index_shape, int64_t index)
{
    std::vector<int64_t> out_shape(stack_shape);
    out_shape.erase(out_shape.begin() + index);
    for(int64_t i=index_shape.size()-1; i>=0; i--)
    {
        out_shape.insert(out_shape.begin() + index,  index_shape[i]);
    }
    return out_shape;
}


// __device__  int64_t* get_coord_by_id_and_stride(int64_t id, int64_t dim int64_t* stride)
// {
//     std::vector<int64_t> coord;
//     int64_t dim = stride.size();
//     int64_t remaining = id;
//     for (int64_t i=0; i<dim; i++)
//     {
//         coord.emplace_back(remaining / stride[i]);
//         remaining -= (remaining / stride[i]) * stride[i];
//     }
//     return coord;
// }

template <int ndim, typename scalar_t>
struct TensorMetadata
{
public:
    int ndim_ = ndim;
    scalar_t* data_ptr ;
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
    int64_t index[ndim];
};

template <int input_ndim, int index_ndim>
__global__ 
void fused_kernel(
        TensorMetadata<input_ndim+index_ndim, float> out_metadata,
        TensorMetadata<input_ndim+1, float> stack_metadata,
        TensorMetadata<index_ndim, int64_t> index_metadata,
        int64_t num_input,
        TensorMetadata<input_ndim, float>* input_metadatas,
        Coord<input_ndim+index_ndim> out_coord,
        Coord<index_ndim> index_coord,
        Coord<input_ndim+1> stack_coord,
        Coord<input_ndim> src_coord,
        Coord<3> add_coord,
        TensorMetadata<3, float> add_metadata,
        int64_t stack_dim,
        int64_t index_dim,
        int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t debug_idx=1447;
    if( idx < numel) {

        // compute out coord
        int64_t remain_index = idx;
        for (int64_t i=0; i<input_ndim+index_ndim; i++)
        {
            int64_t current_dim_index = remain_index/out_metadata.stride[i];
            out_coord.index[i] = current_dim_index;
            remain_index -=  current_dim_index * out_metadata.stride[i];
        }
        if (idx == debug_idx)
        {
        printf("out coord");
        debug_arr(out_coord.index, input_ndim+index_ndim);
        }

        //compute index_coord

        int64_t counter = index_dim;
        for (int64_t i=0; i<index_ndim; i++)
        {
            index_coord.index[i] = out_coord.index[counter];
            counter++;
        }
        if(idx == debug_idx)
        {
        printf("index coord");
        debug_arr(index_coord.index, index_ndim);
        }

        
        // compute index element
        int64_t index_offset = 0;
        for (int64_t i=0; i<index_ndim; i++)
        {
            index_offset += index_coord.index[i] * index_metadata.stride[i];
        }
        int64_t index_element = index_metadata.data_ptr[index_offset];
        if (idx == debug_idx){
        printf("index offset: %d\n", index_offset);
        printf("index element: %d\n", (int)index_element);
        printf("test: %d, %d, %d, %d, %d", index_metadata.data_ptr[0], index_metadata.data_ptr[1],  index_metadata.data_ptr[2],  index_metadata.data_ptr[3],   index_metadata.data_ptr[4]);
        }

        // compute stack coord
        for (int64_t i=0; i<input_ndim+1; i++){
            if (i<index_dim){
                stack_coord.index[i] = out_coord.index[i];
            }
            else if (i==index_dim){
                stack_coord.index[i] =index_element;
            }
            else{
                stack_coord.index[i] = out_coord.index[i+index_ndim-1];
            }
        }
        if (idx == debug_idx)
        {
        printf("stack coord");
        debug_arr(stack_coord.index, input_ndim+1);
        }


        //compute src coord
        int64_t tensor_index = stack_coord.index[stack_dim];
        for (int i=0; i<input_ndim; i++)
        {
            if (i<stack_dim)
            {
                src_coord.index[i] = stack_coord.index[i];
            }
            else if (i>=stack_dim)
            {
                src_coord.index[i] = stack_coord.index[i+1];
            }
        }

        if (idx == debug_idx)
        {
            printf("src coord: ");
            debug_arr(src_coord.index, input_ndim);
        }

        // compute src offset
        int64_t src_offset =0;
        for (int64_t i=0; i<input_ndim; i++){
            src_offset += src_coord.index[i] * input_metadatas[tensor_index].stride[i];
        }

        // copy data
        add_coord.index[0] = 0;
        add_coord.index[1] = stack_coord.index[1];
        add_coord.index[2] = stack_coord.index[3];



        int add_offset = 0;
        for (int i=0; i<3; i++)
        {
            add_offset += add_coord.index[i] * add_metadata.stride[i];
        }

        out_metadata.data_ptr[idx] = input_metadatas[tensor_index].data_ptr[src_offset] + add_metadata.data_ptr[add_offset];
    }
}

// out_coord [u,v,w,x,y,z] -> stack_coord[u, x, y, z] -> src_coord[x, y, z]

template<int input_ndim, int index_ndim>
void launch_kernel(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& add, torch::Tensor& index, int64_t stack_dim, int64_t index_dim)
{
    std::vector<int64_t> index_shape;
    std::vector<int64_t> index_stride;
    for (int64_t i=0; i<index_ndim; i++)
    {
        index_shape.emplace_back(index.size(i));
        index_stride.emplace_back(index.stride(i));
    }
    std::vector<int64_t> stack_shape = get_stack_out_shape(inputs, stack_dim);
    std::vector<int64_t> stack_stride = get_stride_by_shape(stack_shape);
    std::vector<int64_t> out_shape = get_out_shape_by_stack_shape(stack_shape, index_shape, index_dim);
    std::vector<int64_t> out_stride = get_stride_by_shape(out_shape);
    debug_vector<int64_t>("stack_shape", stack_shape);
    debug_vector<int64_t>("stack_stride", stack_stride);
    debug_vector<int64_t>("out_shape", out_shape);
    debug_vector<int64_t>("out_stride", out_stride);
    debug_vector<int64_t>("index_shape", index_shape);
    debug_vector<int64_t>("index_stride", index_stride);
    int64_t numel = out.numel();
    int64_t num_input = inputs.size();

    struct TensorMetadata<input_ndim+index_ndim, float> out_metadata;
    out_metadata.data_ptr = out.data_ptr<float>();
    
    for (int i=0; i<input_ndim+index_ndim; i++)
    {
        out_metadata.stride[i] = out_stride[i];
        out_metadata.size[i] = out_shape[i];
    }

    struct TensorMetadata<index_ndim, int64_t> index_metadata;

    index_metadata.data_ptr = index.data_ptr<int64_t>();
    for (int i=0; i<index_ndim; i++)
    {
        index_metadata.size[i] = index_shape[i];
        index_metadata.stride[i] = index_stride[i];
    }

    struct TensorMetadata<input_ndim+1, float> stack_metadata;
    stack_metadata.data_ptr = nullptr;
    for (int i=0; i<input_ndim+1; i++)
    {
        stack_metadata.stride[i] = stack_stride[i];
        stack_metadata.size[i] = stack_shape[i];
    }

    std::vector<TensorMetadata<input_ndim, float>> input_metadatas;
    for(int i=0; i<num_input; i++)
    {  
        struct TensorMetadata<input_ndim, float> input_meta;
        for (int j=0; j<input_ndim; j++)
        {
            input_meta.stride[j] = inputs[i].stride(j);
            input_meta.size[j] = inputs[i].size(j);
        } 
        input_meta.data_ptr = inputs[i].data_ptr<float>();
        input_metadatas.push_back(input_meta);
    }

    TensorMetadata<input_ndim, float>* in_meta_d;
    cudaMalloc(&in_meta_d, num_input * sizeof(TensorMetadata<input_ndim, float>));
    cudaMemcpy(in_meta_d, input_metadatas.data(), num_input * sizeof(TensorMetadata<input_ndim, float>), cudaMemcpyHostToDevice);
    

    TensorMetadata<3, float> add_metadata;
    add_metadata.data_ptr = add.data_ptr<float>();
    for (int64_t i=0; i<3; i++)
    {
        add_metadata.size[i] = add.size(i);
        add_metadata.stride[i] = add.stride(i);
    }


    struct Coord<input_ndim+index_ndim> out_coord;
    struct Coord<input_ndim+1> stack_coord;
    struct Coord<index_ndim> index_coord;
    struct Coord<input_ndim> src_coord;
    struct Coord<3> add_coord;

    int block = 512;
    int grid = (numel + block -1) /block;

    fused_kernel<input_ndim, index_ndim><<<grid, block>>>(
        out_metadata,
        stack_metadata,
        index_metadata,
        num_input,
        in_meta_d,
        out_coord,
        index_coord,
        stack_coord,
        src_coord,
        add_coord,
        add_metadata,
        stack_dim,
        index_dim,
        numel
        );
    cudaDeviceSynchronize();
    cudaFree(in_meta_d);
}


void fused_stack_add_index_cuda(torch::Tensor& out, const std::vector<torch::Tensor>& inputs, torch::Tensor& add, torch::Tensor& index, int64_t stack_dim, int64_t index_dim){
    int64_t input_ndim = inputs[0].dim();
    int64_t index_ndim = index.dim();
  
    if (input_ndim == 1) 
    {   
        if (index_ndim == 1 )
        {
            launch_kernel<1, 1>(out, inputs, add, index, stack_dim, index_dim);
        }
        else if (index_ndim == 2)
        {
            launch_kernel<1, 2>(out, inputs, add, index, stack_dim, index_dim);
        }
        else
        {
            launch_kernel<1, 2>(out, inputs, add, index, stack_dim, index_dim);

        }
    }
    else if (input_ndim == 2)
    {
        if (index_ndim == 1 )
        {
            launch_kernel<2, 1>(out, inputs, add, index, stack_dim, index_dim);
        }
        else if (index_ndim == 2)
        {
            launch_kernel<2, 2>(out, inputs, add, index, stack_dim, index_dim);
        }
        else
        {
            launch_kernel<2, 2>(out, inputs, add, index, stack_dim, index_dim);

        }   
    }
    else if (input_ndim == 3)
    {
        if (index_ndim == 1 )
        {
            launch_kernel<3, 1>(out, inputs, add, index, stack_dim, index_dim);
        }
        else if (index_ndim == 2)
        {
            launch_kernel<3, 2>(out, inputs, add, index, stack_dim, index_dim);
        }
        else
        {
            launch_kernel<3, 2>(out, inputs, add, index, stack_dim, index_dim);

        }    
    }
    else /* ndim == 4 */
    {
        if (index_ndim == 1 )
        {
            launch_kernel<4, 1>(out, inputs, add, index, stack_dim, index_dim);
        }
        else if (index_ndim == 2)
        {
            launch_kernel<4, 2>(out, inputs, add, index, stack_dim, index_dim);
        }
        else
        {
            launch_kernel<4, 2>(out, inputs, add, index, stack_dim, index_dim);

        }    
    }
}



