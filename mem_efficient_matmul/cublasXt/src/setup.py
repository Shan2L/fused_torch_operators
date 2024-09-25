from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='cublasXt_meffi_mm',
    ext_modules=[
        CUDAExtension(
            name='cublasXt_meffi_mm',
            sources=[
                'cublasXt_meffi_mm.cu',
             ],
            include_dirs=["/usr/local/cuda/include"],
            library_dirs=["/usr/local/cuda/lib64"]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)