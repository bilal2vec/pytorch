import torch

a = torch.ones((8, 8), dtype=torch.bfloat16, device='cuda')
b = torch.ones((8, 8), dtype=torch.bfloat16, device='cuda')
c = torch.ones((8, 8), dtype=torch.bfloat16, device='cuda')

out = torch.addmm(a, b, c)
print(out.shape)
