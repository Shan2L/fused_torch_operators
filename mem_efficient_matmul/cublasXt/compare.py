import torch
import memory_efficient_matmul

a = torch.randn(512, 512).float()
b = torch.randn(512, 512).float()

# cpu_result = torch.mm(a, b)

golden = torch.mm(a.cuda(), b.cuda())

result = memory_efficient_matmul.mm(a, b, 1, 1, 512)

print(result.device)

# print(golden - result)

# equal = torch.isclose(golden, result, 1e-5, 1e-5)
# print(torch.all(equal))
