import torch

a = torch.rand(3, 2, 5)
b = torch.rand(3, 1, 5)


c = a * b
print(c)
