import torch
import memory_efficient_matmul
from torch.profiler import profile, ProfilerActivity





with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_matmul"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:

    a = torch.randn(1024*8, 1024*4).float().cuda()
    b = torch.randn(1024*4, 1024*16).float().cuda()

    golden = torch.mm(a, b)



# print(test.T)