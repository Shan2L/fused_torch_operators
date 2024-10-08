import torch
import cublas_meffi_matmul
torch.set_printoptions(threshold=float('inf'), edgeitems=10)

# a = torch.randn(16, 1024*16, 1024*32).float()
# b = torch.randn(16, 1024*32, 1024*8).float()

a = torch.tensor([item for item in range(0, 240)]).reshape(2, 8, 15).float()
b = torch.tensor([item for item in range(0, 240)]).reshape(2, 15, 8).float()


print(a.shape)
print(b.shape)

# a = torch.ones(2, 4, 4).float()
# b = torch.ones(2, 4, 4).float()

golden = torch.bmm(a.cuda(), b.cuda()).cpu()
res = cublas_meffi_matmul.bmm(a, b, 4)


print(res)

print(f"min: {torch.abs(res-golden).min()}")
print(f"max: {torch.abs(res-golden).max()}")
print(f"mean: {torch.abs(res-golden).mean()}")


with open("golden.txt", 'w')as f1, open("res.txt", "w") as f2:
    f1.write(str(golden))
    f2.write(str(res))