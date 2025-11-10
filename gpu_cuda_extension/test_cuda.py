import torch
import my_kernel

# Array di esempio
N = 1024
a = torch.rand(N, device='cuda')
b = torch.rand(N, device='cuda')

# Somma tramite CUDA kernel
c = my_kernel.add_cuda(a, b)

# Verifica
print(f"a[0] + b[0] = {a[0].item():.4f} + {b[0].item():.4f} = {c[0].item():.4f}")
print("Somma corretta:", torch.allclose(c, a + b))
