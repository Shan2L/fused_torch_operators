import torch
import memory_efficient_matmul

a = torch.randn(512, 512).float()
b = torch.randn(512, 512).float()

cpu_result = torch.mm(a, b)

golden = torch.mm(a.cuda(), b.cuda()).cpu()

result = memory_efficient_matmul.mm(a, b, 1, 1, 128)

equal = torch.isclose(golden, result, 1e-4, 1e-4)
print(torch.all(equal))
