import torch
import cublas_meffi_mm
torch.set_printoptions(threshold=float('inf'), edgeitems=10)

a = torch.randn(1024*16, 1024*32).float()
b = torch.randn(1024*32, 1024*8).float()

res = cublas_meffi_mm.forward(a, b, 1024)
golden = torch.mm(a.cuda(), b.cuda()).cpu()

print(f"min: {torch.abs(res-golden).min()}")
print(f"max: {torch.abs(res-golden).max()}")
print(f"mean: {torch.abs(res-golden).mean()}")


with open("golden.txt", 'w')as f1, open("res.txt", "w") as f2:
    f1.write(str(golden))
    f2.write(str(res))