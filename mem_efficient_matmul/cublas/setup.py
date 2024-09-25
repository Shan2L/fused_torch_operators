from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='memory_efficient_matmul',
    ext_modules=[
        CUDAExtension(
            name='memory_efficient_matmul',
            sources=[
                'single_memory_efficient_matmul.cu',
             ],
            include_dirs=["/usr/local/cuda/include"],
            library_dirs=["/usr/local/cuda/lib64"],
            # extra_compile_args=['-g', '-O0'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)