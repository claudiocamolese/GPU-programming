import torch
import my_cuda_extension

N = 1024
a = torch.rand(N, device='cuda')
b = torch.rand(N, device='cuda')
c = torch.zeros(N, device='cuda')

my_cuda_extension.add_cuda(a, b, c)

if torch.allclose(c, a + b):
    print("CUDA kernel works correctly!")
else:
    print("Mismatch detected!")
