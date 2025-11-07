import torch
import my_kernel

# Array di esempio
N = 1024
a = torch.rand(N, device='cuda')
b = torch.rand(N, device='cuda')

# Somma tramite CUDA kernel
c = my_kernel.add_cuda(a, b)

# Verifica
print("a[0] + b[0] =", a[0].item(), "+", b[0].item(), "=", c[0].item())
print("Somma corretta:", torch.allclose(c, a + b))
