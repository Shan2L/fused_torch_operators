import torch
import cublasXt_meffi_mm
from torch.profiler import profile, ProfilerActivity


with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("../log/cublasXt_meffi_mm"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:

    a = torch.randn(1024*16, 1024*32).float()
    b = torch.randn(1024*32, 1024*8).float()
    result = cublasXt_meffi_mm.forward(a, b, 1, 1, 1024)