from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_kernel',
    ext_modules=[
        CUDAExtension(
            'my_kernel',
            ['my_kernel_wrapper.cpp', 'my_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '-gencode=arch=compute_75,code=sm_75',  # architettura Turing
                    '--use_fast_math'
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
