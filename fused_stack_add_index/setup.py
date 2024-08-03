from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_stack_add_index',
    ext_modules=[
        CUDAExtension(
            name='fused_stack_add_index',
            sources=[
                'fused_stack_add_index.cpp',
                'fused_kernel.cu'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)