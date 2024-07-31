import torch
import fused_stack_add
from torch.profiler import profile, ProfilerActivity

with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./fused_stack_add"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:
    tensor_list = []
    for i in range(10):
        tensor_list.append(torch.randn((48, 512, 1024)).cuda());

    weight = torch.randn((1, 10, 1024), dtype=torch.float).cuda()

    fused_out = fused_stack_add.forward(tensor_list, weight, 0)


