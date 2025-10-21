from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='my_cuda_extension',
            sources=['my_kernel.cu', 'my_kernel_wrapper.cpp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
