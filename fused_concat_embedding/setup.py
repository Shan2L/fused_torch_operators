from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_cat_embedding',
    ext_modules=[
        CUDAExtension(
            name='fused_cat_embedding',
            sources=[
                'fused_cat_embedding.cpp',
                'fused_kernel.cu'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)