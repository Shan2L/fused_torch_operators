import torch
import memory_efficient_matmul
from torch.profiler import profile, ProfilerActivity

# with profile(
# 	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
# 	on_trace_ready=torch.profiler.tensorboard_trace_handler("./efficient_mamtul_512_mem_cpy_del"),
# 	profile_memory=True,
# 	with_stack=True,
# 	record_shapes=True,
#     use_cuda=True) as prof:
torch.set_printoptions(threshold=float('inf'), edgeitems=10)


a = torch.randn(2016, 1000).float().cuda()
b = torch.randn(1000, 4000).float().cuda()

print(a.stride())
print(b.stride())

res = memory_efficient_matmul.mm(a.cpu(), b.cpu(), 512).cuda()
golden = torch.matmul(a, b)

print(f"min: {torch.abs(res-golden).min()}")
print(f"max: {torch.abs(res-golden).max()}")
print(f"mean: {torch.abs(res-golden).mean()}")


with open("golden.txt", 'w')as f1, open("res.txt", "w") as f2:
    f1.write(str(golden))
    f2.write(str(res))