from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_stack_add',
    ext_modules=[
        CUDAExtension(
            name='fused_stack_add',
            sources=[
                'fused_stack_add.cpp',
                'fused_kernel.cu'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)