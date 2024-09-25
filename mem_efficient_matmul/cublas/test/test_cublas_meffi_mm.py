import torch
import cublas_meffi_mm
from torch.profiler import profile, ProfilerActivity
# torch.set_printoptions(threshold=float('inf'), edgeitems=10)

with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("../log/cublas_meffi_mm"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:

    a = torch.randn(1024*16, 1024*32).float()
    b = torch.randn(1024*32, 1024*8).float()
    res = cublas_meffi_mm.forward(a, b, 1024)
