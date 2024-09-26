from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='cublas_meffi_matmul',
    ext_modules=[
        CUDAExtension(
            name='cublas_meffi_matmul',
            sources=[
                'cublas_meffi_mm.cu',
                'cublas_meffi_bmm.cu',
                'cublas_meffi_matmul.cpp'
             ],
            include_dirs=["/usr/local/cuda/include"],
            library_dirs=["/usr/local/cuda/lib64"],
            extra_compile_args=['-g', '-O0'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)