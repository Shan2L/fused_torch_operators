import torch
import cublasXt_meffi_mm

a = torch.randn(512, 512).float()
b = torch.randn(512, 512).float()

# cpu_result = torch.mm(a, b)

golden = torch.mm(a.cuda(), b.cuda())

result = cublasXt_meffi_mm.forward(a, b, 1, 1, 512)

print(result.device)

print(golden - result)

equal = torch.isclose(golden, result, 1e-5, 1e-5)
print(torch.all(equal))
