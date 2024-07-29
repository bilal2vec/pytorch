import torch

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda')
y = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32, device='cuda')

def fun(x):
    return x + 1

fun = torch.compile(fun, mode='reduce-overhead')

print('warmup')
out = fun(x)
print('compile')
out = fun(x)
print('replay')
out = fun(x)

print(out)

x[0] = 10.0

out = fun(x)

print(out)

out = fun(y)

print(out)